
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import json
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

import vertexai
from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('matching_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FounderInvestorMatcher:
    def __init__(self, max_workers: int = 3, batch_size: int = 2):
        """Initialize with conservative defaults to stay within quotas"""
        try:
            self.project_id = "your-project-id"
            self.location = "us-central1"
            
            # Initialize Vertex AI with explicit project/location
            vertexai.init(
                project=self.project_id,
                location=self.location
            )
            
            # Use stable model version
            self.model = GenerativeModel("gemini-2.5-pro-exp-03-25")
            
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            self.max_workers = max_workers
            self.batch_size = batch_size
            self.min_funding_score = 0.3
            self.top_matches_limit = 200  # Reduced for quota management
            
            # Rate limiting controls
            self.api_call_count = 0
            self.last_api_call_time = 0
            self.min_call_interval = 1.2  # Seconds between API calls
            self.max_calls_per_minute = 30
            
            logger.info(f"Initialized matcher with {max_workers} workers and batch size {batch_size}")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _enforce_rate_limit(self):
        """Enforce strict rate limits with jitter to avoid quota issues"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call_time
        
        # If we're making calls too fast
        if elapsed < self.min_call_interval:
            sleep_time = self.min_call_interval - elapsed + random.uniform(0.1, 0.3)
            time.sleep(sleep_time)
        
        # Reset counter if we've passed a minute
        if elapsed > 60:
            self.api_call_count = 0
            
        # If approaching limit, slow down further
        if self.api_call_count >= self.max_calls_per_minute * 0.9:
            time.sleep(random.uniform(2.0, 3.0))
            
        self.api_call_count += 1
        self.last_api_call_time = time.time()

    def preprocess_data(self, founders_df: pd.DataFrame, investors_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clean and prepare data for matching"""
        try:
            # Basic cleaning
            founders_df = founders_df.dropna(subset=['industry', 'stage', 'funding_required']).copy()
            investors_df = investors_df.dropna(subset=['Industry', 'Stage', 'Cheque_range']).copy()

            # Create combined text features
            founders_df['combined_features'] = founders_df.apply(
                lambda row: (
                    f"Industry:{row['industry'].lower().strip()} "
                    f"Stage:{row['stage'].lower().strip()} "
                    f"Funding:{self._normalize_funding_str(row['funding_required'])} "
                    f"Model:{str(row.get('business_model', '')).lower()}"
                ),
                axis=1
            )

            investors_df['combined_preferences'] = investors_df.apply(
                lambda row: (
                    f"Industry:{str(row['Industry']).lower().strip()} "
                    f"Stage:{str(row['Stage']).lower().strip()} "
                    f"Range:{self._normalize_funding_str(row['Cheque_range'])} "
                    f"Type:{str(row.get('Type', '')).lower()}"
                ),
                axis=1
            )

            # Fit TF-IDF vectorizer
            all_text = pd.concat([founders_df['combined_features'], investors_df['combined_preferences']])
            self.vectorizer.fit(all_text)

            return founders_df, investors_df
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def get_batch_insights(self, batch_pairs: List[Tuple[Dict, Dict]]) -> List[Optional[Dict]]:
        """Get insights for a batch of matches with strict rate limiting"""
        try:
            self._enforce_rate_limit()
            
            prompt = """Analyze these founder-investor pairs. For each, provide:
            1. Compatibility score (1-10)
            2. Top 3 alignment factors
            3. Any potential red flags
            
            Format each analysis as:
            ### Pair {n} ###
            - Founder: {founder_name}
            - Investor: {investor_name}
            - Score: X/10
            - Alignment: 
              - Factor 1
              - Factor 2
              - Factor 3
            - Concerns:
              - Concern 1 (if any)
            
            Here are the pairs:\n"""
            
            for i, (founder, investor) in enumerate(batch_pairs):
                prompt += f"\n### Pair {i+1} ###\n"
                prompt += f"- Founder: {founder['name']} ({founder['industry']}, {founder['stage']})\n"
                prompt += f"- Investor: {investor['Name']} ({investor['Industry']}, {investor['Stage']})\n"
                prompt += f"- Funding Needed: {founder['funding_required']} vs Range: {investor['Cheque_range']}\n"

            response = self.model.generate_content(
                contents=[prompt],
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_output_tokens": 4000
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH
                }
            )

            return self._parse_batch_response(response.text, batch_pairs)
        except Exception as e:
            logger.error(f"Batch insight generation failed: {str(e)}")
            # Implement exponential backoff
            wait_time = min(2 ** (self.api_call_count // 10), 60) + random.uniform(0, 1)
            time.sleep(wait_time)
            return [None] * len(batch_pairs)

    def _parse_batch_response(self, response_text: str, batch_pairs: List[Tuple[Dict, Dict]]) -> List[Optional[Dict]]:
      """Parse the API response into structured data with improved extraction"""
      try:
          analyses = []
          current_analysis = {}
          current_section = None
          
          for line in response_text.split('\n'):
              line = line.strip()
              
              # Start of a new pair analysis
              if line.startswith('### Pair'):
                  if current_analysis:
                      analyses.append(current_analysis)
                  current_analysis = {
                      'alignment_factors': [],
                      'concerns': []
                  }
                  current_section = None
                  
              # Score line
              elif line.lower().startswith('- score:'):
                  try:
                      score_part = line.split(':')[1].strip()
                      score = float(score_part.split('/')[0].strip())
                      current_analysis['insight_score'] = score
                  except (IndexError, ValueError):
                      current_analysis['insight_score'] = 0
                      
              # Section headers
              elif line.lower().startswith('- alignment:'):
                  current_section = 'alignment'
              elif line.lower().startswith('- concerns:'):
                  current_section = 'concerns'
                  
              # Content lines (factors/concerns)
              elif line.startswith('-') and current_section:
                  content = line[1:].strip()
                  if current_section == 'alignment' and content:
                      current_analysis['alignment_factors'].append(content)
                  elif current_section == 'concerns' and content:
                      current_analysis['concerns'].append(content)
                      
          # Add the last analysis if exists
          if current_analysis:
              analyses.append(current_analysis)

          # Ensure we have enough analyses
          if len(analyses) < len(batch_pairs):
              logger.warning(f"Only got {len(analyses)} analyses for {len(batch_pairs)} pairs")
              # Pad with empty analyses if needed
              analyses.extend([{'alignment_factors': [], 'concerns': []} for _ in range(len(batch_pairs) - len(analyses))])
              
          results = []
          for i, (pair, analysis) in enumerate(zip(batch_pairs, analyses)):
              founder, investor = pair
              result = {
                  'founder_id': founder['id'],
                  'founder_name': founder['name'],
                  'investor_id': investor['Name'],
                  'investor_name': investor['Name'],
                  'match_score': analysis.get('insight_score', 0) * 10,
                  'alignment_factors': analysis.get('alignment_factors', []),
                  'concerns': analysis.get('concerns', []),
                  'insights': f"Pair {i+1} analysis:\n" + response_text[:500]  # Store first 500 chars
              }
              results.append(result)
          
          return results
      except Exception as e:
          logger.error(f"Batch response parsing failed: {str(e)}")
          # Return empty analyses for all pairs if parsing fails
          return [{
              'founder_id': pair[0]['id'],
              'founder_name': pair[0]['name'],
              'investor_id': pair[1]['Name'],
              'investor_name': pair[1]['Name'],
              'match_score': 0,
              'alignment_factors': [],
              'concerns': [],
              'insights': "Analysis parsing failed"
          } for pair in batch_pairs]

    def _process_numeric_match(self, founder: Dict, investor: Dict) -> Optional[Dict]:
        """Calculate numeric compatibility scores (no API calls)"""
        try:
            fund_score = self._check_funding_compatibility(
                founder['funding_required'],
                investor['Cheque_range']
            )
            
            if fund_score < self.min_funding_score:
                return None
                
            stage_score = self._check_stage_compatibility(
                founder['stage'],
                investor['Stage']
            )
            
            industry_score = self._check_industry_compatibility(
                founder['industry'],
                investor['Industry']
            )
            
            # Text similarity
            founder_vec = self.vectorizer.transform([founder['combined_features']])
            investor_vec = self.vectorizer.transform([investor['combined_preferences']])
            text_sim = cosine_similarity(founder_vec, investor_vec)[0][0]
            
            # Weighted final score (0-100)
            final_score = (
                0.4 * fund_score + 
                0.3 * stage_score + 
                0.2 * industry_score + 
                0.1 * text_sim
            ) * 100
            
            return {
                'founder_id': founder['id'],
                'founder_name': founder['name'],
                'investor_id': investor['Name'],
                'investor_name': investor['Name'],
                'match_score': round(final_score, 2),
                'funding_score': round(fund_score * 100, 2),
                'stage_score': round(stage_score * 100, 2),
                'industry_score': round(industry_score * 100, 2),
                'text_similarity': round(text_sim * 100, 2),
                'investor_type': investor.get('Type', ''),
                'investor_industry': investor.get('Industry', ''),
                'investor_stage': investor.get('Stage', ''),
                'investor_funding_range': investor.get('Cheque_range', ''),
                'investor_linkedin': investor.get('Linkedin_Personal', ''),
                'investor_twitter': investor.get('Twitter', '')
            }
        except Exception as e:
            logger.warning(f"Numeric match failed for {founder.get('name')}-{investor.get('Name')}: {str(e)}")
            return None

    def calculate_matches(self, founders_df: pd.DataFrame, investors_df: pd.DataFrame) -> List[Dict]:
        """Orchestrate the entire matching process with rate limiting"""
        try:
            logger.info("Starting matching process...")
            numeric_results = []
            total_pairs = len(founders_df) * len(investors_df)
            
            # Phase 1: Numeric matching (fast, no API calls)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_numeric_match, founder.to_dict(), investor.to_dict())
                    for _, founder in founders_df.iterrows()
                    for _, investor in investors_df.iterrows()
                ]
                
                with tqdm(total=total_pairs, desc="Numeric matching") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            numeric_results.append(result)
                        pbar.update(1)

            # Phase 2: Qualitative analysis (rate limited API calls)
            logger.info(f"Processing qualitative insights for top {self.top_matches_limit} matches...")
            insights_results = []
            top_matches = sorted(numeric_results, key=lambda x: x['match_score'], reverse=True)[:self.top_matches_limit]

            # Prepare batches
            batches = []
            current_batch = []
            
            for match in top_matches:
                founder = founders_df[founders_df['id'] == match['founder_id']].iloc[0].to_dict()
                investor = investors_df[investors_df['Name'] == match['investor_id']].iloc[0].to_dict()
                current_batch.append((founder, investor))
                
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch.copy())
                    current_batch = []
            
            if current_batch:
                batches.append(current_batch)

            # Process batches with strict rate limiting
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
                insight_futures = {
                    executor.submit(self.get_batch_insights, batch): i
                    for i, batch in enumerate(batches)
                }
                
                for future in tqdm(as_completed(insight_futures), total=len(batches), desc="Getting insights"):
                    batch_results = future.result()
                    for result in batch_results:
                        if result:
                            # Merge with numeric scores
                            numeric_match = next(
                                (m for m in top_matches if 
                                 m['founder_id'] == result['founder_id'] and 
                                 m['investor_id'] == result['investor_id']),
                                None
                            )
                            if numeric_match:
                                merged = {**numeric_match, **result}
                                insights_results.append(merged)

            # Combine results (insights first, then numeric only)
            final_results = insights_results + [
                m for m in numeric_results 
                if m not in top_matches
            ]
            
            return final_results
        except Exception as e:
            logger.error(f"Matching process failed: {str(e)}")
            raise

    def _save_intermediate_results(self, results: List[Dict], filename: str = "interim_matches.csv"):
        """Periodically save results to avoid data loss"""
        try:
            pd.DataFrame(results).to_csv(filename, index=False)
            logger.info(f"Interim results saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save interim results: {str(e)}")

    def _normalize_funding_str(self, funding_str: str) -> str:
        """Standardize funding range formatting"""
        return re.sub(r'[\$,]', '', funding_str).upper().replace(' ', '')

    def _check_funding_compatibility(self, founder_needs: str, investor_range: str) -> float:
        """Calculate funding compatibility score (0-1)"""
        try:
            def parse_range(s):
                s = self._normalize_funding_str(s)
                if '-' in s:
                    parts = s.split('-')
                    return float(parts[0].replace('K', '000').replace('M', '000000')), \
                           float(parts[1].replace('K', '000').replace('M', '000000'))
                val = float(s.replace('K', '000').replace('M', '000000'))
                return val, val

            f_min, f_max = parse_range(founder_needs)
            i_min, i_max = parse_range(investor_range)

            if f_max < i_min or f_min > i_max:
                return 0.0

            overlap_min = max(f_min, i_min)
            overlap_max = min(f_max, i_max)
            overlap = overlap_max - overlap_min
            f_range = f_max - f_min

            return min(overlap / f_range, 1.0) if f_range > 0 else 0.0
        except Exception as e:
            logger.warning(f"Funding compatibility error: {str(e)}")
            return 0.0

    def _check_stage_compatibility(self, founder_stage: str, investor_stages: str) -> float:
        """Calculate stage compatibility score (0-1)"""
        try:
            founder_stage = founder_stage.lower().strip()
            investor_stages = investor_stages.lower()
            
            stage_mapping = {
                'pre-seed': 0,
                'seed': 1,
                'series a': 2,
                'series b': 3,
                'series c': 4,
                'series+': 2,
                'growth': 5
            }
            
            founder_val = stage_mapping.get(founder_stage, -1)
            if founder_val == -1:
                return 0.5
                
            investor_vals = []
            for stage in investor_stages.split(','):
                stage = stage.strip()
                val = stage_mapping.get(stage, -1)
                if val != -1:
                    investor_vals.append(val)
            
            if not investor_vals:
                return 0.5
                
            if founder_val in investor_vals:
                return 1.0
                
            if any(abs(founder_val - iv) <= 1 for iv in investor_vals):
                return 0.7
                
            return 0.3
        except Exception as e:
            logger.warning(f"Stage compatibility error: {str(e)}")
            return 0.5

    def _check_industry_compatibility(self, founder_industry: str, investor_industry: str) -> float:
        """Calculate industry compatibility score (0-1)"""
        try:
            founder_industry = founder_industry.lower().strip()
            investor_industry = investor_industry.lower().strip()
            
            if 'agnostic' in investor_industry:
                return 1.0
                
            if founder_industry == investor_industry:
                return 1.0
                
            # Check for keyword overlap
            founder_keywords = set(re.findall(r'\w+', founder_industry))
            investor_keywords = set(re.findall(r'\w+', investor_industry))
            common = founder_keywords & investor_keywords
            
            if not common:
                return 0.0
                
            return 0.8 if len(common) >= 2 else 0.5
        except Exception as e:
            logger.warning(f"Industry compatibility error: {str(e)}")
            return 0.5

def load_data(json_path: str, founder_sample: int = None, investor_sample: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and optionally sample data"""
    try:
        with open(json_path) as f:
            investor_data = json.load(f)
        
        if investor_sample:
            investor_data = investor_data[:investor_sample]
            
        investors_df = pd.DataFrame(investor_data)
        
        # Sample founder data
        founders_data = [
            {
                'id': 'F001',
                'name': 'SecureSaaS',
                'industry': 'Information Technology & Services',
                'stage': 'Seed',
                'funding_required': '$1M-$3M',
                'traction': '50 enterprise customers',
                'business_model': 'B2B SaaS'
            },
            {
                'id': 'F002',
                'name': 'EduTechAI',
                'industry': 'Edtech',
                'stage': 'Pre-seed',
                'funding_required': '$200K-$500K',
                'traction': 'Pilot with 5 schools',
                'business_model': 'B2B2C Subscription'
            },
            {
                'id': 'F003',
                'name': 'WikiAnalytics',
                'industry': 'Sector Agnostic',
                'stage': 'Series+',
                'funding_required': '$5M-$8M',
                'traction': '100K monthly active users',
                'business_model': 'Freemium'
            }
        ]
        
        if founder_sample:
            founders_data = founders_data[:founder_sample]
            
        founders_df = pd.DataFrame(founders_data)
        
        logger.info(f"Loaded {len(investors_df)} investors and {len(founders_df)} founders")
        return founders_df, investors_df
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def main():
    try:
        # Initialize with conservative settings
        matcher = FounderInvestorMatcher(max_workers=3, batch_size=2)
        
        # Load data with small samples for testing
        print("\nLoading data...")
        founders_df, investors_df = load_data('data.json', founder_sample=2, investor_sample=50)
        
        # Preprocess data
        print("Preprocessing data...")
        founders_df, investors_df = matcher.preprocess_data(founders_df, investors_df)
        
        # Calculate matches
        print("\nCalculating matches (this may take a while due to rate limiting)...")
        start_time = time.time()
        matches = matcher.calculate_matches(founders_df, investors_df)
        duration = (time.time() - start_time) / 60
        print(f"\nMatching completed in {duration:.2f} minutes")
        
        # Save results
        output_file = 'founder_investor_matches.csv'
        pd.DataFrame(matches).to_csv(output_file, index=False)
        print(f"\nSaved {len(matches)} matches to {output_file}")
        
        # Display top matches
        print("\nGenerating reports...")
        for founder_id in founders_df['id'].unique():
            founder_name = founders_df[founders_df['id'] == founder_id]['name'].values[0]
            founder_matches = [m for m in matches if m['founder_id'] == founder_id and 'insights' in m]
            top_matches = sorted(founder_matches, key=lambda x: x['match_score'], reverse=True)[:3]
            
            print(f"\n=== Top matches for {founder_name} ===")
            for i, match in enumerate(top_matches, 1):
                print(f"\n#{i}: {match['investor_name']} (Score: {match['match_score']:.1f})")
                print(f"  Funding: {match['funding_score']:.1f} | Stage: {match['stage_score']:.1f} | Industry: {match['industry_score']:.1f}")
                print(f"  LinkedIn: {match.get('investor_linkedin', 'N/A')}")
                print(f"  Twitter: {match.get('investor_twitter', 'N/A')}")
                print("\n  Key Insights:")
                for factor in match.get('alignment_factors', [])[:3]:
                    print(f"  - {factor}")
                if match.get('concerns'):
                    print("\n  Potential Concerns:")
                    for concern in match['concerns'][:2]:
                        print(f"  - {concern}")
                
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
