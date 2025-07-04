import json
import logging
import os
from typing import Dict, Any

import requests

from my_proof.models.proof_response import ProofResponse


class Proof:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.proof_response = ProofResponse(dlp_id=config['dlp_id'])

    def generate(self) -> ProofResponse:
        """Generate proofs for all input files."""
        logging.info("Starting proof generation")

        # Iterate through files and calculate data validity
        account_email = None

        for input_filename in os.listdir(self.config['input_dir']):
            input_file = os.path.join(self.config['input_dir'], input_filename)
            if os.path.splitext(input_file)[1].lower() == '.json':
                with open(input_file, 'r') as f:
                    input_data = json.load(f)

                    if input_filename == 'account.json':
                        account_email = input_data.get('email', None)
                        logging.info(f"Account email: {account_email}")
                        continue

                    elif input_filename == 'Saved Places.json':
                        # --- New logic for Saved Places.json ---
                        features = input_data.get('features', [])
                        num_places = len(features)
                        required_fields = ['geometry', 'properties']
                        required_props = ['date', 'google_maps_url', 'location']
                        required_loc = ['address', 'country_code', 'name']
                        complete_count = 0
                        coords = []
                        for feat in features:
                            # Check completeness
                            if all(k in feat for k in required_fields):
                                props = feat['properties']
                                loc = props.get('location', {})
                                if all(k in props for k in required_props) and all(k in loc for k in required_loc):
                                    complete_count += 1
                            # Collect coordinates
                            if 'geometry' in feat and 'coordinates' in feat['geometry']:
                                coords.append(tuple(feat['geometry']['coordinates']))
                        completeness = complete_count / num_places if num_places > 0 else 0
                        # Calculate max pairwise distance (Haversine)
                        import math
                        def haversine(coord1, coord2):
                            lon1, lat1 = coord1
                            lon2, lat2 = coord2
                            R = 6371  # Earth radius in km
                            phi1, phi2 = math.radians(lat1), math.radians(lat2)
                            dphi = math.radians(lat2 - lat1)
                            dlambda = math.radians(lon2 - lon1)
                            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
                            return 2*R*math.asin(math.sqrt(a))
                        max_dist = 0
                        for i in range(len(coords)):
                            for j in range(i+1, len(coords)):
                                dist = haversine(coords[i], coords[j])
                                if dist > max_dist:
                                    max_dist = dist
                        # Normalize metrics (stricter)
                        min_places = 2  # Minimum places for nonzero score
                        max_places = 50  # Need 50+ for max score
                        max_possible_dist = 20000  # half Earth's circumference in km
                        norm_num_places = 0 if num_places < min_places else min((num_places - min_places) / (max_places - min_places), 1.0)
                        norm_completeness = completeness
                        # Make dispersion more important and stricter
                        min_dispersion_km = 2  # Need at least 2km for nonzero
                        norm_dispersion = 0 if max_dist < min_dispersion_km else min((max_dist - min_dispersion_km) / (max_possible_dist - min_dispersion_km), 1.0)
                        # Weighted score: more weight to dispersion
                        quality = round(0.6 * norm_completeness + 0.2 * norm_num_places + 0.2 * norm_dispersion, 3)
                        uniqueness = round(0.5 * norm_num_places + 0.5 * norm_dispersion, 3)
                        self.proof_response.quality = quality
                        self.proof_response.uniqueness = uniqueness
                        self.proof_response.authenticity = 0
                        self.proof_response.score = 0.7 * quality + 0.3 * uniqueness 
                        self.proof_response.valid = num_places >= min_places and completeness > 0.8 and max_dist >= min_dispersion_km
                        if not hasattr(self.proof_response, 'attributes') or self.proof_response.attributes is None:
                            self.proof_response.attributes = {}
                        self.proof_response.attributes.update({
                            'completeness': completeness,
                            'quality': quality,
                            'uniqueness': uniqueness,
                        })
                        self.proof_response.metadata = {
                            'dlp_id': self.config['dlp_id'],
                        }

        logging.info(f"Account email in config: {self.config['user_email']}")
        email_matches = self.config['user_email'] == account_email
        score_threshold = 0.2

        # Calculate proof-of-contribution scores: https://docs.vana.org/vana/core-concepts/key-elements/proof-of-contribution/example-implementation
        #self.proof_response.ownership = 1.0 if email_matches else 0.0 
        self.proof_response.ownership = 0 #for now we are not checking ownership

        # Additional (public) properties to include in the proof about the data
        if not hasattr(self.proof_response, 'attributes') or self.proof_response.attributes is None:
            self.proof_response.attributes = {}
        self.proof_response.attributes.update({
            'score_threshold': score_threshold,
            'email_verified': self.proof_response.ownership == 1.0,
        })

        # Additional metadata about the proof, written onchain
        self.proof_response.metadata = {
            'dlp_id': self.config['dlp_id'],
        }

        return self.proof_response