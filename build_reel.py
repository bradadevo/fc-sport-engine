import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import librosa
from PIL import Image

def get_audio_beats(audio_path):
    """Analyze audio to find transient beats using librosa."""
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}, returning default intervals.")
        return [0.0, 2.0, 4.0, 6.0, 8.0]
    
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beats = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    return beats.tolist()

def generate_fcpxml(image_paths, beats, output_path="reel.fcpxml"):
    """Generates an FCPXML 1.11 project with Ken Burns effect."""
    root = ET.Element("fcpxml", version="1.11")
    
    # Resources
    resources = ET.SubElement(root, "resources")
    ET.SubElement(resources, "format", id="r1", name="FFVideoFormat1080p30", frameDuration="100/3000s", width="1080", height="1920")
    
    # Add image resources
    for i, img in enumerate(image_paths):
        ET.SubElement(resources, "asset", id=f"img{i}", name=f"Image {i}", src=f"file://{os.path.abspath(img)}")

    # Sequence
    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="Sports Reel Event")
    project = ET.SubElement(event, "project", name="Sports Reel")
    sequence = ET.SubElement(project, "sequence", format="r1", tcStart="0s", tcFormat="NDF")
    spine = ET.SubElement(sequence, "spine")
    
    fps = 30
    current_time = 0.0
    
    # Create clips based on beats
    for i in range(len(image_paths)):
        start_time = beats[i] if i < len(beats) else current_time
        end_time = beats[i+1] if i + 1 < len(beats) else start_time + 2.0
        duration = end_time - start_time
        
        # Clip
        clip = ET.SubElement(spine, "asset-clip", 
                             ref=f"img{i}",
                             name=f"Image {i}",
                             offset=f"{int(current_time * fps)}00/3000s",
                             duration=f"{int(duration * fps)}00/3000s",
                             format="r1")
        
        # Ken Burns (Video Animation)
        video = ET.SubElement(clip, "video", name=f"Image {i}", offset="0s")
        transform = ET.SubElement(video, "adjust-transform", scale="1.2 1.2")
        
        current_time += duration
        
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n')
        f.write(xmlstr[xmlstr.find('<fcpxml'):])
    
    print(f"Generated {output_path}")

def main():
    # Placeholder for running the logic
    print("Sports Reel Engine Initialized.")
    # Example logic:
    # beats = get_audio_beats("input_audio.wav")
    # generate_fcpxml(["photo1.jpg", "photo2.jpg"], beats)

if __name__ == "__main__":
    main()