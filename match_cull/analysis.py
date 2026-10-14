import difflib

from PIL import Image
from pydantic import BaseModel, Field

from sports_reel.vision import generate_structured, image_to_jpeg

CACHE_FILENAME = ".match_cull_analysis.json"

PROMPT = """You are culling photos from a youth soccer match for a sports photographer.

Score this photo from 1 to 10 as a keeper for the highlight gallery:
- 9-10: peak action (ball contact, save, tackle, header, goal celebration), tack-sharp on the main
  player, faces visible, clean composition.
- 6-8: good, usable action or emotion with minor flaws.
- 4-5: weak: soft focus, awkward moment, subject small or partly blocked.
- 1-3: reject: blurry, missed focus, eyes closed, back of heads only, empty field, accidental frame.

Also return:
- reason: one short sentence explaining the score.
- subject_description: what is happening.
- the bounding box (normalized 0.0-1.0) of the main subject: the player on the ball, the point of
  contact, or the celebrating group.
- the best crop, as a skilled sports photo editor would frame it: keep the action, the ball and
  faces; trim distracting edges (empty grass, spectators, poles, cut-off limbs); leave room in the
  direction of play; use the rule of thirds. Give it as crop_xmin/crop_ymin/crop_xmax/crop_ymax
  (normalized 0.0-1.0) and pick crop_shape, the aspect ratio that suits the shot, from:
  4:5, 1:1, 3:2, 2:3, 16:9, 9:16. If the full frame is already best, return the whole photo.
- best_look and second_look: the two best color looks for this photo, chosen ONLY from this list
  (use the exact names). Consider light (day, night, overcast), the kit colors and the mood.
{looks}
"""


class PhotoAssessment(BaseModel):
    score: int = Field(description="Keeper score from 1 (reject) to 10 (portfolio)")
    reason: str = Field(description="One short sentence explaining the score")
    subject_description: str = Field(description="What is happening in the photo")
    ymin: float = Field(description="Normalized top edge of the main subject 0.0-1.0")
    xmin: float = Field(description="Normalized left edge of the main subject 0.0-1.0")
    ymax: float = Field(description="Normalized bottom edge of the main subject 0.0-1.0")
    xmax: float = Field(description="Normalized right edge of the main subject 0.0-1.0")
    crop_xmin: float = Field(description="Normalized left edge of the best crop 0.0-1.0")
    crop_ymin: float = Field(description="Normalized top edge of the best crop 0.0-1.0")
    crop_xmax: float = Field(description="Normalized right edge of the best crop 0.0-1.0")
    crop_ymax: float = Field(description="Normalized bottom edge of the best crop 0.0-1.0")
    crop_shape: str = Field(description="Aspect ratio of the best crop: 4:5, 1:1, 3:2, 2:3, 16:9 or 9:16")
    best_look: str = Field(description="Best color look, exactly as named in the list")
    second_look: str = Field(description="Second-best color look, different from best_look")


TASTE_SECTION = """
This photographer's own taste, learned from their favorite photos. Weigh it when scoring
(a moment they would love can score well even without ball action):
{notes}
"""


def build_prompt(look_descriptions: dict[str, str], taste_notes: str = "") -> str:
    if look_descriptions:
        lines = "\n".join(f"- {name}: {desc}" for name, desc in look_descriptions.items())
    else:
        lines = "- (no looks installed: answer 'none' for both)"
    prompt = PROMPT.format(looks=lines)
    if taste_notes:
        prompt += TASTE_SECTION.format(notes=taste_notes)
    return prompt


def match_look(name: str, available: list[str]) -> str | None:
    """Maps the model's answer to an installed look name, tolerating case and
    small spelling differences."""
    if not available:
        return None
    lowered = {a.lower(): a for a in available}
    if name.strip().lower() in lowered:
        return lowered[name.strip().lower()]
    close = difflib.get_close_matches(name.strip().lower(), list(lowered), n=1, cutoff=0.75)
    return lowered[close[0]] if close else None


def resolve_looks(assessment: PhotoAssessment, available: list[str]) -> tuple[str | None, str | None]:
    """The two looks to use, always valid and different when possible. A
    look the model invented is replaced by the first installed one."""
    if not available:
        return None, None
    best = match_look(assessment.best_look, available) or available[0]
    second = match_look(assessment.second_look, available)
    if not second or second == best:
        second = next((a for a in available if a != best), best)
    return best, second


def analyze(client, img: Image.Image, look_descriptions: dict[str, str], model: str,
            usage: list | None = None, taste_notes: str = "") -> PhotoAssessment:
    assessment = generate_structured(client, image_to_jpeg(img), build_prompt(look_descriptions, taste_notes),
                                     PhotoAssessment, model=model, usage=usage)
    assessment.score = max(1, min(10, assessment.score))
    return assessment
