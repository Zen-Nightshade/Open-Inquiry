from .base_parser import BaseParser
import json
import re

class JSONEnforcer(BaseParser):
    
    def __init__(self, output_schema=None):
        """
        schema: dict defining expected keys and types
        Example:
        {
            "name": str,
            "age": int
        }
        """
        self.schema = output_schema

    def parse(self, text):
        data = self._extract_json(text)
        self._validate_json(data)

        if self.schema:
            self._enforce_schema(data)

        return data

    def _extract_json(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

    def _validate_json(self, data):
        if not isinstance(data, dict):
            raise ValueError("JSON must be an object (dict)")

    def _enforce_schema(self, data):
        for key, expected_type in self.schema.items():
            if key not in data:
                raise ValueError(f"Missing key: {key}")

            if not isinstance(data[key], expected_type):
                raise ValueError(
                    f"Key '{key}' must be of type {expected_type.__name__}"
                )