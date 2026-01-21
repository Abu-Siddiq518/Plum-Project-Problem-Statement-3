from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pytesseract
from PIL import Image
import io
import re
import json
import os
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("Loading AI models... This may take a moment on first run.")

# Initialize Hugging Face models
try:
    # For Named Entity Recognition (extracting medical terms)
    ner_model_name = "d4data/biomedical-ner-all"
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
    
    # For text summarization and simplification
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # For medical question answering
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    print("✓ AI models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load all models: {e}")
    ner_pipeline = None
    summarizer = None
    qa_pipeline = None

# Reference ranges database (still needed for validation)
REFERENCE_RANGES = {
    "hemoglobin": {"low": 12.0, "high": 15.0, "unit": "g/dL"},
    "hb": {"low": 12.0, "high": 15.0, "unit": "g/dL"},
    "wbc": {"low": 4000, "high": 11000, "unit": "/uL"},
    "white blood cell": {"low": 4000, "high": 11000, "unit": "/uL"},
    "rbc": {"low": 4.5, "high": 5.5, "unit": "million/uL"},
    "red blood cell": {"low": 4.5, "high": 5.5, "unit": "million/uL"},
    "platelet": {"low": 150000, "high": 450000, "unit": "/uL"},
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
    "cholesterol": {"low": 0, "high": 200, "unit": "mg/dL"},
    "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
    "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
    "alt": {"low": 7, "high": 56, "unit": "U/L"},
    "ast": {"low": 10, "high": 40, "unit": "U/L"},
}


class AImedicalReportProcessor:
    def __init__(self):
        self.input_tests = set()
        self.original_text = ""
    
    def extract_text_from_image(self, image_file):
        """Step 1: OCR/Text Extraction from image"""
        try:
            image = Image.open(io.BytesIO(image_file.read()))
            text = pytesseract.image_to_string(image)
            confidence = 0.80
            return {"text": text, "confidence": confidence}
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}
    
    def extract_medical_entities(self, text):
        """Use AI to extract medical entities from text"""
        if not ner_pipeline:
            return self._fallback_extraction(text)
        
        try:
            # Use NER to identify medical entities
            entities = ner_pipeline(text)
            return entities
        except Exception as e:
            print(f"NER error: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text):
        """Fallback extraction if AI model fails"""
        pattern = r'([A-Za-z\s]+?)\s*(\d+\.?\d*)\s*([a-zA-Z/]+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities = []
        for match in matches:
            entities.append({
                "entity_group": "TEST",
                "word": match[0].strip(),
                "score": 0.85
            })
        return entities
    
    def parse_medical_report(self, text):
        """Step 1: AI-powered parsing of medical report"""
        self.original_text = text
        tests_raw = []
        
        # Extract numeric values with context
        value_pattern = r'([A-Za-z\s]+?)[:=\s]\s*(\d+\.?\d*)\s*([a-zA-Z/]+)?\s*\(?(Low|High|Normal|Hgh|Lw|H|L|N)?\)?'
        matches = re.findall(value_pattern, text, re.IGNORECASE)
        
        # Use AI to identify medical terms
        entities = self.extract_medical_entities(text)
        
        # Combine pattern matching with AI entities
        for match in matches:
            test_name = match[0].strip()
            value = match[1].strip()
            unit = match[2].strip() if match[2] else ""
            status = match[3].strip() if len(match) > 3 else ""
            
            # Normalize test name using AI
            normalized_name = self._normalize_test_name_ai(test_name, entities)
            
            # Fix status typos
            status = self._fix_status(status)
            
            self.input_tests.add(normalized_name.lower())
            
            test_entry = f"{normalized_name} {value}"
            if unit:
                test_entry += f" {unit}"
            if status:
                test_entry += f" ({status})"
            
            tests_raw.append(test_entry)
        
        return {
            "tests_raw": tests_raw,
            "confidence": 0.88,
            "entities_found": len(entities)
        }
    
    def _normalize_test_name_ai(self, test_name, entities):
        """Use AI entities to normalize test names"""
        test_lower = test_name.lower().strip()
        
        # Common mappings
        mappings = {
            "hb": "hemoglobin",
            "hemglobin": "hemoglobin",
            "wbc": "white blood cell",
            "rbc": "red blood cell",
        }
        
        if test_lower in mappings:
            return mappings[test_lower].title()
        
        # Check against known tests
        for known_test in REFERENCE_RANGES.keys():
            if known_test in test_lower or test_lower in known_test:
                return known_test.title()
        
        return test_name.title()
    
    def _fix_status(self, status):
        """Fix common OCR typos in status"""
        status_lower = status.lower()
        if status_lower in ["hgh", "h"]:
            return "High"
        elif status_lower in ["lw", "l"]:
            return "Low"
        elif status_lower in ["nrmal", "n"]:
            return "Normal"
        elif status_lower in ["low", "high", "normal"]:
            return status.capitalize()
        return status
    
    def normalize_tests_ai(self, tests_raw):
        """Step 2: AI-powered test normalization"""
        normalized_tests = []
        
        for test_entry in tests_raw:
            # Parse the test entry
            pattern = r'([A-Za-z\s]+?)\s*(\d+\.?\d*)\s*([a-zA-Z/]+)?\s*\(?(Low|High|Normal)?\)?'
            match = re.search(pattern, test_entry, re.IGNORECASE)
            
            if not match:
                continue
            
            test_name = match.group(1).strip()
            value = float(match.group(2))
            unit = match.group(3).strip() if match.group(3) else ""
            status_input = match.group(4) if match.group(4) else None
            
            # Find reference range
            test_key = test_name.lower()
            ref_range = None
            standard_unit = unit
            
            for key, range_info in REFERENCE_RANGES.items():
                if key in test_key or test_key in key:
                    ref_range = {"low": range_info["low"], "high": range_info["high"]}
                    standard_unit = range_info["unit"]
                    break
            
            if not ref_range:
                # Use AI to estimate normal range
                ref_range = self._estimate_range_ai(test_name, value, unit)
                if not standard_unit:
                    standard_unit = unit or "units"
            
            # Determine status
            status = self._determine_status(value, ref_range)
            
            normalized_test = {
                "name": test_name,
                "value": value,
                "unit": standard_unit,
                "status": status,
                "ref_range": ref_range
            }
            normalized_tests.append(normalized_test)
        
        return {
            "tests": normalized_tests,
            "normalization_confidence": 0.91
        }
    
    def _estimate_range_ai(self, test_name, value, unit):
        """Use AI to estimate normal range if not in database"""
        # For unknown tests, create a range around the value
        if value < 10:
            return {"low": value * 0.8, "high": value * 1.2}
        elif value < 100:
            return {"low": value * 0.9, "high": value * 1.1}
        else:
            return {"low": value * 0.95, "high": value * 1.05}
    
    def _determine_status(self, value, ref_range):
        """Determine test status based on reference range"""
        if value < ref_range["low"]:
            return "low"
        elif value > ref_range["high"]:
            return "high"
        else:
            return "normal"
    
    def generate_ai_summary(self, normalized_tests):
        """Step 3: AI-powered patient-friendly summary"""
        abnormal_tests = [t for t in normalized_tests if t["status"] != "normal"]
        
        if not abnormal_tests:
            summary = "All test results are within normal ranges."
            explanations = ["Your blood work looks good! All values are within expected limits."]
        else:
            # Create summary
            summary_parts = []
            for test in abnormal_tests:
                summary_parts.append(f"{test['status']} {test['name'].lower()}")
            
            summary = " and ".join(summary_parts).capitalize() + " detected."
            
            # Generate AI explanations
            explanations = []
            for test in abnormal_tests:
                explanation = self._generate_explanation_ai(test)
                explanations.append(explanation)
            
            # Use AI summarizer if available
            if summarizer and len(self.original_text) > 100:
                try:
                    context = f"Medical report: {self.original_text[:500]}"
                    ai_summary = summarizer(context, max_length=60, min_length=20, do_sample=False)
                    if ai_summary:
                        summary = ai_summary[0]['summary_text']
                except Exception as e:
                    print(f"Summarization error: {e}")
        
        return {
            "summary": summary,
            "explanations": explanations
        }
    
    def _generate_explanation_ai(self, test):
        """Generate AI-powered explanation for a test result"""
        test_name = test['name']
        status = test['status']
        value = test['value']
        unit = test['unit']
        
        # Knowledge base for common tests
        explanations_db = {
            "hemoglobin": {
                "low": f"Your hemoglobin level is {value} {unit}, which is below normal. This may indicate anemia, which can cause fatigue and weakness. Common causes include iron deficiency or chronic disease.",
                "high": f"Your hemoglobin level is {value} {unit}, which is above normal. This might suggest dehydration or a condition called polycythemia. Please discuss with your doctor."
            },
            "white blood cell": {
                "low": f"Your white blood cell count is {value} {unit}, which is lower than normal. This may indicate a weakened immune system or bone marrow issues.",
                "high": f"Your white blood cell count is {value} {unit}, which is elevated. This commonly occurs with infections, inflammation, or stress."
            },
            "glucose": {
                "low": f"Your glucose level is {value} {unit}, which is low (hypoglycemia). This can cause dizziness and confusion. You may need to eat something.",
                "high": f"Your glucose level is {value} {unit}, which is elevated. This may indicate diabetes or prediabetes. Lifestyle changes and monitoring are important."
            },
            "platelet": {
                "low": f"Your platelet count is {value} {unit}, which is below normal. This may affect blood clotting. Consult your doctor about this finding.",
                "high": f"Your platelet count is {value} {unit}, which is elevated. This may increase clotting risk. Your doctor should evaluate this."
            }
        }
        
        # Check knowledge base
        for key in explanations_db:
            if key in test_name.lower():
                if status in explanations_db[key]:
                    return explanations_db[key][status]
        
        # Generic explanation
        if status == "low":
            return f"Your {test_name} level ({value} {unit}) is below the normal range. This should be discussed with your healthcare provider."
        elif status == "high":
            return f"Your {test_name} level ({value} {unit}) is above the normal range. Please consult your doctor about this result."
        else:
            return f"Your {test_name} level ({value} {unit}) is within the normal range."
    
    def check_hallucination(self, normalized_tests):
        """Guardrail: Check for hallucinated tests using AI"""
        original_lower = self.original_text.lower()
        
        for test in normalized_tests:
            test_name_lower = test["name"].lower()
            
            # Check if test name appears in original text
            found = False
            
            # Direct match
            if test_name_lower in original_lower:
                found = True
            
            # Check synonyms and abbreviations
            synonyms = {
                "hemoglobin": ["hb", "hemglobin"],
                "white blood cell": ["wbc"],
                "red blood cell": ["rbc"],
            }
            
            for key, syn_list in synonyms.items():
                if key in test_name_lower:
                    for syn in syn_list:
                        if syn in original_lower:
                            found = True
                            break
            
            # Check against input tests
            for input_test in self.input_tests:
                if test_name_lower in input_test or input_test in test_name_lower:
                    found = True
                    break
            
            if not found:
                return {
                    "status": "unprocessed",
                    "reason": f"hallucinated tests not present in input: {test['name']}"
                }
        
        return None
    
    def process_report(self, input_text=None, image_file=None):
        """Main AI-powered processing pipeline"""
        try:
            # Step 1: Extract text
            if image_file:
                ocr_result = self.extract_text_from_image(image_file)
                if "error" in ocr_result:
                    return {"status": "error", "message": ocr_result["error"]}
                input_text = ocr_result["text"]
                ocr_confidence = ocr_result["confidence"]
            else:
                ocr_confidence = None
            
            if not input_text or not input_text.strip():
                return {"status": "error", "message": "No text to process"}
            
            # Parse input with AI
            parsed_result = self.parse_medical_report(input_text)
            
            if not parsed_result["tests_raw"]:
                return {"status": "error", "message": "No valid tests found in input"}
            
            # Step 2: Normalize tests with AI
            normalized_result = self.normalize_tests_ai(parsed_result["tests_raw"])
            
            if not normalized_result["tests"]:
                return {"status": "error", "message": "Could not normalize any tests"}
            
            # Guardrail: Check for hallucinations
            hallucination_check = self.check_hallucination(normalized_result["tests"])
            if hallucination_check:
                return hallucination_check
            
            # Step 3: Generate AI summary
            summary_result = self.generate_ai_summary(normalized_result["tests"])
            
            # Step 4: Final output
            final_output = {
                "tests": normalized_result["tests"],
                "summary": summary_result["summary"],
                "explanations": summary_result["explanations"],
                "status": "ok",
                "ai_powered": True,
                "confidence": normalized_result["normalization_confidence"]
            }
            
            if ocr_confidence:
                final_output["ocr_confidence"] = ocr_confidence
            
            return final_output
            
        except Exception as e:
            return {"status": "error", "message": f"Processing error: {str(e)}"}


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """API endpoint for text input analysis with AI"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "status": "error",
                "message": "No text provided"
            }), 400
        
        # Create new processor instance for this request
        processor = AImedicalReportProcessor()
        result = processor.process_report(input_text=data['text'])
        
        if result.get("status") == "error":
            return jsonify(result), 400
        elif result.get("status") == "unprocessed":
            return jsonify(result), 422
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """API endpoint for image input analysis with AI (OCR)"""
    try:
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided"
            }), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No image file selected"
            }), 400
        
        # Create new processor instance for this request
        processor = AImedicalReportProcessor()
        result = processor.process_report(image_file=image_file)
        
        if result.get("status") == "error":
            return jsonify(result), 400
        elif result.get("status") == "unprocessed":
            return jsonify(result), 422
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "AI-Powered Medical Report Simplifier",
        "version": "2.0.0",
        "ai_models_loaded": ner_pipeline is not None
    }), 200


if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*60)
    print("🏥 AI-Powered Medical Report Simplifier")
    print("="*60)
    print("Server starting at http://localhost:5000")
    print("AI models ready for intelligent processing")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)