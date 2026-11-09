from .decode_config import DecodeConfig

class Preset():
    @staticmethod
    def qa_preset() -> DecodeConfig:
        return DecodeConfig(
            temperature=0.2,
            top_p=0.9,
            top_k=0,
            do_sample=True,
            max_tokens=256,
            repetition_penalty=1.05,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            citation_required=True,
        )

    @staticmethod
    def rewrite_preset() -> DecodeConfig:
        return DecodeConfig(
            temperature=0.6,
            top_p=0.95,
            top_k=0,
            do_sample=True,
            max_tokens=1024,
            repetition_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.2,
        )

    @staticmethod
    def summary_preset() -> DecodeConfig:
        return DecodeConfig(
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            max_tokens=512,
            length_penalty=1.2,
            repetition_penalty=1.05,
        )

    @staticmethod
    def survey_preset() -> DecodeConfig:
        return DecodeConfig(
            temperature=0.5,
            top_p=0.95,
            do_sample=True,
            max_tokens=2048,
            repetition_penalty=1.15,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            citation_required=True,
        )
