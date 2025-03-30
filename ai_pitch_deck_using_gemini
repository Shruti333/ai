

import os
import re
import json
import logging
from typing import Dict, List
from google import genai
from google.genai import types
from google.colab import files

import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PitchDeckAnalyzer:
    SECTION_WEIGHTS = {
        'problem': 0.20,
        'solution': 0.25,
        'market': 0.15,
        'business_model': 0.20,
        'financials': 0.10,
        'team': 0.10
    }

    EVALUATION_CRITERIA = {
        'problem': ["Clarity of problem definition", "Target audience specificity", "Problem magnitude quantification"],
        'solution': ["Innovation level", "Feasibility", "Differentiation from competitors"],
        'market': ["Market size quantification", "Growth potential", "Target segment clarity"],
        'business_model': ["Revenue streams clarity", "Scalability", "Customer acquisition strategy"],
        'financials': ["Realistic projections", "Unit economics", "Funding needs justification"],
        'team': ["Relevant expertise", "Complementary skills", "Track record"]
    }

    def __init__(self, project_id: str = "your-project-key", location: str = "us-central1"):
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        self.model_name = "gemini-2.5-pro-exp-03-25"

    def _generate_content(self, prompt: str) -> str:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]

        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return response.text

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\x0c', '', text)
        text = re.sub(r'[^\w\s.,;:!?\-%\$\u20AC\u00A3]', '', text)
        return text.strip()

    def split_into_sections_with_gemini(self, full_text: str) -> Dict[str, str]:
      prompt = f"""
      You are a pitch deck evaluator AI. Read the following startup pitch content deeply and properly and extract the following sections properly . Identify each section even if given briefly as mentioned and look for the context :
        - Problem: Any mention of challenges, pain points, or needs being addressed
        - Solution: Any description of products, services, or approaches to solve the problem
        - Market: Any discussion of customers, industry size, trends, or target segments
        - Business Model: Any information about revenue, pricing, monetization, or go-to-market
        - Financials: Any numbers about costs, revenue, funding, or projections
        - Team: Any mention of founders, key personnel, or advisors
    
      Return ONLY a valid JSON object with these exact keys:
      'problem', 'solution', 'market', 'business_model', 'financials', 'team'.

      Do not provide explanations or any text outside the JSON format.

      Content:
      '''{full_text}'''
      """

      try:
          response = self._generate_content(prompt)
          match = re.search(r'\{[\s\S]*\}', response)
          if match:
              json_text = match.group(0)
              return json.loads(json_text)
          else:
              raise ValueError("Could not extract valid JSON.")
      except Exception as e:
          logger.error(f"Gemini section extraction failed: {e}")
          return {key: "" for key in self.SECTION_WEIGHTS.keys()}


    def analyze_section(self, section_name: str, section_text: str) -> Dict:
        if not section_text.strip():
            return {
                'section': section_name,
                'score': 0,
                'strengths': ["Section not found"],
                'weaknesses': ["This critical section is missing"],
                'suggestions': [f"Add a detailed {section_name.replace('_', ' ')} section"],
                'evaluation': {}
            }

        prompt = f"""
        Analyze the '{section_name}' section from a startup pitch deck deeply and properly amd score them accordingly.Note how much have they defined each section annd even if the definition is less be lenient and give them some marks instead of 0  . Give 0 only in the case when that section is entirely not talked about .:
        Criteria: {', '.join(self.EVALUATION_CRITERIA[section_name])}

        Section Content:
        {section_text}

        Format:
        Score: [0-100]
        Be generous while giving scores
        Strengths:
        - [strength 1]
        Weaknesses:
        - [weakness 1]
        Suggestions:
        - [suggestion 1]
        Evaluation:
        - [criterion 1]: [score]/5 - [comment]
        """

        try:
            response = self._generate_content(prompt)
            return self._parse_analysis_response(section_name, section_text, response)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'section': section_name,
                'score': 0,
                'strengths': [],
                'weaknesses': [f"Analysis error: {str(e)}"],
                'suggestions': [],
                'evaluation': {}
            }

    def _parse_analysis_response(self, section_name: str, section_text: str, response: str) -> Dict:
        result = {
            'section': section_name,
            'text': section_text[:500] + '...' if len(section_text) > 500 else section_text,
            'score': 0,
            'strengths': [],
            'weaknesses': [],
            'suggestions': [],
            'evaluation': {}
        }

        try:
            score_match = re.search(r'Score:\s*(\d{1,3})', response)
            if score_match:
                result['score'] = min(100, int(score_match.group(1)))

            result['strengths'] = re.findall(r'Strengths:\s*-\s*(.*)', response)
            result['weaknesses'] = re.findall(r'Weaknesses:\s*-\s*(.*)', response)
            result['suggestions'] = re.findall(r'Suggestions:\s*-\s*(.*)', response)

            eval_matches = re.findall(r'-\s*(.*?):\s*(\d)/5\s*-\s*(.*)', response)
            for crit, score, comment in eval_matches:
                result['evaluation'][crit.strip()] = {
                    'score': int(score),
                    'comment': comment.strip()
                }

        except Exception as e:
            logger.error(f"Error parsing response: {e}")

        return result

    def calculate_overall_score(self, section_analyses: List[Dict]) -> float:
        total_score = 0
        total_weight = 0
        for analysis in section_analyses:
            weight = self.SECTION_WEIGHTS.get(analysis['section'], 0)
            total_score += analysis['score'] * weight
            total_weight += weight
        return round(total_score / total_weight if total_weight else 0, 1)

    def generate_executive_summary(self, analyses: List[Dict], overall_score: float) -> str:
        prompt = """Create an executive summary of this pitch deck analysis . Do not use section words amd be brief .:

Overall Score: {overall_score}/100

Key Findings:
{section_findings}

Provide:
1. Top 3 strengths
2. Top 3 weaknesses
3. Investment Readiness (High/Medium/Low)
4. Recommendations
"""

        section_findings = []
        for analysis in analyses:
            section_findings.append(
                f"{analysis['section'].upper()} ({analysis['score']}/100)\nStrengths: {', '.join(analysis['strengths'][:2])}\nImprovements: {', '.join(analysis['weaknesses'][:2])}"
            )

        try:
            return self._generate_content(prompt.format(
                overall_score=overall_score,
                section_findings='\n\n'.join(section_findings)
            ))
        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    def analyze_pitch_deck(self, text: str) -> Dict:
        clean_text = self.preprocess_text(text)
        sections = self.split_into_sections_with_gemini(clean_text)
        analyses = []
        for section_name, section_text in sections.items():
            logger.info(f"Analyzing {section_name} section...")
            analysis = self.analyze_section(section_name, section_text)
            analyses.append(analysis)
        overall_score = self.calculate_overall_score(analyses)
        summary = self.generate_executive_summary(analyses, overall_score)
        return {
            'overall_score': overall_score,
            'section_analyses': analyses,
            'executive_summary': summary
        }

def extract_text_from_pdf(file_path: str) -> str:
    text = extract_text_with_pymupdf(file_path)
    if not text.strip():
        return extract_text_with_ocr(file_path)
    return text

def extract_text_with_pymupdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logger.warning(f"Failed PyMuPDF extraction: {e}")
        return ""

def extract_text_with_ocr(file_path: str) -> str:
    try:
        pages = convert_from_path(file_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
        return text
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""

def main():
    print("=== Pitch Deck Analyzer ===")
    print("1. Paste pitch text\n2. Upload text or PDF")
    choice = input("Choose (1/2): ")

    if choice == '1':
        pitch_text = input("Paste your pitch text:\n")
    else:
        uploaded = files.upload()
        file_name = next(iter(uploaded))
        if file_name.endswith(".pdf"):
            pitch_text = extract_text_from_pdf(file_name)
        else:
            with open(file_name, 'r') as f:
                pitch_text = f.read()

    analyzer = PitchDeckAnalyzer()
    print("\nAnalyzing pitch deck...")
    results = analyzer.analyze_pitch_deck(pitch_text)

    print(f"\n\n\u2b50 Overall Score: {results['overall_score']}/100")
    print("\n\U0001F50D Section Scores:")
    for analysis in results['section_analyses']:
        print(f"{analysis['section'].title()} - {analysis['score']}")
        strength = analysis['strengths'][0] if analysis['strengths'] else 'N/A'
        weakness = analysis['weaknesses'][0] if analysis['weaknesses'] else 'N/A'
        print(f"  Strength: {strength}\n  Weakness: {weakness}")

    print("\n\U0001F4DD Executive Summary:")
    print(results['executive_summary'])

    if input("\nSave report? (y/n): ").lower() == 'y':
        with open('pitch_analysis_report.txt', 'w') as f:
            f.write(f"Overall Score: {results['overall_score']}\n\n")
            for analysis in results['section_analyses']:
                f.write(f"{analysis['section'].title()} ({analysis['score']}):\n")
                f.write("Strengths:\n" + '\n'.join(f"- {s}" for s in analysis['strengths']) + '\n')
                f.write("Weaknesses:\n" + '\n'.join(f"- {w}" for w in analysis['weaknesses']) + '\n')
                f.write("Suggestions:\n" + '\n'.join(f"- {s}" for s in analysis['suggestions']) + '\n')
                f.write("Evaluation:\n" + '\n'.join(f"- {k}: {v['score']}/5 - {v['comment']}" for k, v in analysis['evaluation'].items()) + '\n\n')
            f.write("Executive Summary:\n" + results['executive_summary'])
        print("\nSaved to pitch_analysis_report.txt")

if __name__ == "__main__":
    main()
