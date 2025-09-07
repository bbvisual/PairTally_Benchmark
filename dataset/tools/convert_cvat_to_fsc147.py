#!/usr/bin/env python3
"""
Conversion script for CVAT annotations to FSC147 format.
Converts parsed_annotations.json (CVAT format) to:
1. annotation_FSC147_384.json (all images)
2. Train_Test_Val_FSC_147.json (all images)
3. annotation_FSC147_384_inter.json (INTER objects only)
4. Train_Test_Val_FSC_147_inter.json (INTER objects only)
5. annotation_FSC147_384_intra.json (INTRA objects only)
6. Train_Test_Val_FSC_147_intra.json (INTRA objects only)

Optional (with --create_super_category_files flag):
7. annotation_FSC147_384_{category}.json (per super category: HOU, OTR, FOO, etc.)
8. Train_Test_Val_FSC_147_{category}.json (per super category: HOU, OTR, FOO, etc.)
"""

import json
import argparse
import os
import random
import re
import hashlib

# Object description correction dictionary
OBJECT_DESCRIPTIONS = {
    "FOO": {
        "PAS": [
            "Curved beige pieces with a smooth surface and slight ridges.",
            "Short, twisted yellow spirals with ridged textures.",
            "Hollow cylinders with angled ends and a pale yellow tone."
        ],
        "RIC": [
            "Tiny white grains with a semi-translucent sheen and pointed ends.",
            "Small, slightly translucent grains with a narrow shape and glossy white finish.",
            "Tiny, elongated brown kernels with a matte texture and fibrous look."
        ],
        "LIM": [
            "Small round green fruits with dimpled skin and a glossy finish.",
            "Spherical objects with smooth dark green skin and a citrusy texture.",
            "Tiny round green fruits with shiny skin and firm texture, smaller than typical citrus."
        ],
        "PEP": [
            "Tiny black spheres with a rough, dry surface and matte appearance.",
            "Dark, wrinkled spheres with hard exteriors and uneven textures.",
            "Small white or off-white balls with a chalky surface and faint speckles."
        ],
        "TOM": [
            "Smooth red spheres with glossy skin and plump form.",
            "Round fruits with soft red skin and moderate size.",
            "Miniature round fruits with shiny skin, often in clusters and vibrant red or orange."
        ],
        "CHI": [
            "Long, slender red vegetables with curved tips and glossy finish.",
            "Extended vegetables with smooth red skin and pointed ends.",
            "Shorter curved items with glossy skin, red."
        ],
        "PNT": [
            "Oval, light brown nuts with a wrinkled skin and a dry finish.",
            "Pale brown seeds with a reddish outer skin and rough texture.",
            "Beige kernels with a smooth surface and no skin covering."
        ],
        "BEA": [
            "Smooth oval items with uniform color like white or tan.",
            "Glossy black oval legumes with firm texture and small white spot.",
            "Soft, round seeds with a pale yellow finish and smooth surface."
        ],
        "SED": [
            "Flat green seeds with slightly curved edges and a smooth surface.",
            "Hard-shelled oval seeds, usually dark green and matte.",
            "Small, striped or white flat seeds with a glossy texture."
        ],
        "CFC": [
            "Rectangular candies in wrappers with sharp edges and glossy.",
            "Shiny rectangular sweets in lighter brown shades.",
            "Rectangular black candies with glossy wrappers."
        ],
        "ONI": [
            "Small round bulbs with purple outer skins and layered structure."
        ],
        "CAN": [
            "Rectangular candies in wrappers with sharp edges and glossy."
        ],
        "GAR": [
            "Bulbous white segments clustered together with a papery peel."
        ]
    },

    "FUN": {
        "CHK": [
            "Flat black discs with a smooth, plastic surface and uniform thickness.",
            "Circular flat pieces with a matte black finish and ridged edges.",
            "White round discs with polished surfaces and slightly raised edges."
        ],
        "MAH": [
            "Rectangular ivory-colored tiles with engraved symbols and glossy finish.",
            "Flat white pieces with bamboo patterns in green and red paint.",
            "Shiny rectangular tiles with red geometric chinese characters on smooth surface."
        ],
        "LEG": [
            "Small plastic circles.",
            "Small green plastic circles.",
            "Small pink plastic circles."
        ],
        "CHS": [
            "Polished black figurines with classic shapes like towers or knights.",
            "Glossy, sculpted figures in dark color with intricate detailing.",
            "Bright white figures with smooth bases and detailed tops, often symmetrical."
        ],
        "PZP": [
            "Cardboard pieces with interlocking edges and printed image fragments.",
            "Flat pieces with straight edges and smooth corners, used for borders.",
            "Central jigsaw parts with wavy outlines."
        ],
        "PUZ": [
            "Cardboard pieces with interlocking edges and printed image fragments.",
            "Flat pieces with straight edges and smooth corners, used for borders.",
            "Central jigsaw parts with wavy outlines."
        ],
        "PKC": [
            "Flat plastic discs with circular ridges.",
            "Round blue tokens with smooth, matte surface and printed center.",
            "White plastic circles with thick edges and centered logos."
        ],
        "PLC": [
            "Flexible rectangular items with smooth surfaces and colorful back sides.",
            "Glossy rectangular cards with red decorations and smooth edges.",
            "Thin cards with black symbols on white background and curved corners.",
        ],
        "MAR": [
            "Glass spheres with colorful swirls inside and shiny transparent surface.",
            "Larger orbs with thick glass walls and intricate internal coloring.",
            "Tiny translucent marbles with soft hues and smooth texture."
        ],
        "DIC": [
            "Cube-shaped items with engraved dots on each face and clean edges.",
            "Green plastic cubes with black pips and glossy surface.",
            "White dice with black dots and slightly rounded corners."
        ],
        "CSC": [
            "Thin rectangular ivory cards with a matte finish and no markings.",
            "Slim cards with glossy white surface and black visible symbols.",
            "Ivory cards with small red prints and a smooth, thin profile."
        ]
    },

    "HOU": {
        "TPK": [
            "Thin, white sticks with pointed ends and a slight grain texture.",
            "Straight white rods with rounded tips and a glossy surface.",
            "Thin flexible sticks with dual-colored tips and a plastic strand."
        ],
        "CTB": [
            "Short sticks with fluffy white ends.",
            "Light brown shafts with cotton padding on both sides.",
            "White plastic rods with rounded white cotton at each end."
        ],
        "PIL": [
            "Small smooth tablets in pure white, often oval or round.",
            "Flat circular items with a white matte finish and score lines.",
            "Rounded yellow capsules with glossy surface and slightly raised center."
        ],
        "BAT": [
            "Short silver cylinders with branded wrappers and flat ends.",
            "Thin cylindrical objects with a metallic coating and rounded tops.",
            "Larger cylinders with a wider diameter and color-coded labels."
        ],
        "HCP": [
            "Dark tools with curved edges and a shiny plastic finish.",
            "Matte black objects with narrow heads and ridged texture.",
            "Brown clippers with a sleek surface."
        ],
        "MNY": [
            "Colorful rectangular notes with intricate printed designs and numbers.",
            "Pale grey sheets of thin paper with fine details and text.",
            "Blue notes with fine details and text."
        ],
        "COI": [
            "Circular metal discs with ridged edges and stamped patterns.",
            "Smaller flat silver items with geometric engravings and shiny finish.",
            "Larger round, matte-finish metallic tokens with smooth edges."
        ],
        "BOT": [
            "Round caps used for bottles.",
            "Round metallic caps with crimped edges and embossed branding.",
            "Black flat plastic lids with grooved sides and simple textures."
        ],
        "BBT": [
            "Plastic discs with symmetrical hole patterns and smooth texture.",
            "Flat, circular fasteners with four tiny holes and raised edges.",
            "Rounded fasteners with two centered holes and glossy finish."
        ],
        "ULT": [
            "Lightweight white utensils.",
            "Long-handled tool with a wide, rounded end and white surface.",
            "Narrow eating tool with sharp prongs and smooth clear plastic body."
        ]
    },
    "OFF": {
        "PPN": [
            "Tiny fastening tool with a sharp metal point and a round plastic head, available in a wide range of bright colors and a smooth glossy finish.",
            "Standard fastener with a sharp tip and a square-shaped plastic head, featuring clean edges and vibrant solid colors like red, blue, or yellow.",
            "Compact round-headed pin with a shiny surface, made of colored plastic and paired with a short metallic shaft for easy pinning."
        ],
        "HST": [
            "Shiny stickers shaped like hearts.",
            "Large heart-shaped decals with glossy finish.",
            "Miniature glittery stickers shaped like hearts."
        ],
        "CRS": [
            "Colored rectangular sticks with soft edges and matte finish.",
            "Thin wooden sticks with rounded ends and bright red-orange color.",
            "Flat craft sticks dyed in deep blue or purple, with smooth texture."
        ],
        "RUB": [
            "Thin rubber loops with smooth finishes and varying sizes.",
            "Looped stretchy bands in soft yellow and a matte feel.",
            "Flexible circular bands with a slightly glossy blue surface."
        ],
        "STN": [
            "Sticky sheets with smooth writing surface.",
            "Rectangular paper pads in dark green shades with adhesive edges.",
            "Light green notes with clean lines and soft paper texture."
        ],
        "PPC": [
            "Colorful clips with bent ends and a polished surface.",
            "Small bent wires in bright colors, used for clamping paper.",
            "Shiny metallic holders with a silver sheen and oval shape."
        ],
        "PEN": [
            "Long cylinder with a pointed end.",
            "Plastic writing tool with a detachable cap and narrow tip.",
            "Smooth writing instrument without a cap and tapered point."
        ],
        "PNC": [
            "Wooden cylindrical writing tool with a dark core inside."
        ],
        "RHS": [
            "Shiny beads in reflective.",
            "Bright stones with clear sparkle and round shape.",
            "Tiny flat gems shaped like stars with metallic glint."
        ],
        "ZPT": [
            "Thin plastic strips with jagged teeth and locking tips.",
            "Short, bendable fasteners with a textured grip surface.",
            "Long white ties with ribbed surface and a square end."
        ],
        "SFP": [
            "Curved metallic pins with a clasp and a coiled spring end.",
            "Large safety pins with a dull silver finish and safety loop.",
            "Tiny pins with rounded ends and a simple locking clasp."
        ],
        "LPP": [
            "Tiny decorative pins with a shiny surface and secure clasp.",
        ],
        "WWO": [
            "Plastic strip with loops for bundling wires, transparent."
        ]
    },
    "OTR": {
        "SCR": [
            "Short, coiled fastener with a metallic body and flat or pointed tip, commonly used in construction tasks.",
            "Long, silver-colored fastener with deep helical grooves and a sharp point, designed for hard surfaces.",
            "Compact bronze-toned piece with a threaded shaft and blunt tip, used for tight, short-distance fastening."
        ],
        "BOL": [
            "Thick metal fastener with a smooth shaft and flat end, usually paired with a matching ring to secure components.",
            "Short metallic shaft with a six-sided top and fine threading, designed for secure tightening.",
            "Fastening piece with a rounded, dome-like head and thick cylindrical body, giving it a smooth, button-like top."
        ],
        "NUT": [
            "Small metallic ring with internal threading, designed to pair with other fasteners for locking.",
            "Hexagonal fastening piece with a sturdy, six-sided shape and a central screw opening.",
            "Square-shaped ring with internal threading and flat edges for easy gripping."
        ],
        "WAS": [
            "Flat, circular item with a hole in the center, used to distribute pressure across surfaces.",
            "Thin metal ring with a shiny, silver surface and a smooth feel, typically placed under fasteners.",
            "Lightweight, white ring made of plastic with a soft matte texture and minimal thickness."
        ],
        "BUT": [
            "Small, round fastener with multiple holes in the center, usually matte with soft edges and subtle color.",
            "Circular item in a muted beige tone, slightly textured and commonly used for fastening fabric.",
            "Glossy transparent disc with smooth edges and centered holes, creating a barely-there appearance."
        ],
        "NAI": [
            "Thin metallic rod with a flat top and pointed tip, smooth along the shaft with no threading.",
            "Medium-length silver piece with a wide flat head and smooth tapered body, designed for general use.",
            "Short, thick pin with a textured gray finish and strong tapered end, used in masonry or dense materials."
        ],
        "BEA": [
            "Tiny, decorative item with a central hole for threading, often made of glass or plastic and used in crafts.",
            "Round or oval-shaped ornament in deep blue and purple hues, with a glossy polished finish.",
            "Smooth, spherical or cylindrical piece in bright orange and pink shades, often used in playful or decorative settings."
        ],
        "IKC": [
            "Plastic closure tool with a hinge mechanism, flat on one end and ridged along the inner gripping edge.",
            "Green-colored clamping piece with a textured plastic surface and a hinged design for sealing bags or bundles.",
            "Bright red version of the same gripping device, featuring a flat handle and firm internal teeth for securing contents."
        ],
        "IKE": [
            "Compact snapping tool with a plastic hinge and ridged edges, designed to clamp and hold items in place.",
            "Green variant of the device with a textured surface and sturdy clasp, ideal for sealing.",
            "Red variation featuring a flat clamp profile and a firm lock, made of slightly glossy plastic."
        ],
        "PEG": [
            "Spring-loaded clamping device with two long arms and a pressure point, often used for hanging items.",
            "Gray-toned gripping tool with a strong hinge, wide mouth, and ribbed holding surfaces.",
            "White clamp with a sleek, minimalist design and a strong grip, commonly used in domestic settings."
        ],
        "STO": [
            "Irregular, naturally formed object with a rough matte surface and deep red earthy tones.",
            "Textured stone with jagged edges and a slightly grainy red finish, often darker in tone.",
            "Rounded or uneven object in a dusty yellow hue with a coarse surface and natural speckles."
        ]
    }

}


# {
#     "FOO": {
#         "PAS": ["pasta", "spiral pasta", "penne pasta"],
#         "RIC": ["rice grain", "jasmine rice grain", "brown rice grain"],
#         "LIM": ["citrus fruit", "lime", "calamansi"],
#         "PEP": ["peppercorn", "black peppercorn", "white peppercorn"],
#         "TOM": ["tomato", "normal tomato", "baby tomato"],
#         "CHI": ["chili", "long chili", "short chili"],
#         "PNT": ["peanut", "peanut with skin", "peanut without skin"],
#         "BEA": ["bean", "black bean", "soy bean"],
#         "SED": ["seed", "pumpkin seed", "sunflower seed"],
#         "CFC": ["coffee candy", "brown coffee candy", "black coffee candy"],
#         "ONI": ["shallot"],
#         "CAN": ["candy"],
#         "GAR": ["garlic"]
#     },
#     "FUN": {
#         "CHK": ["checker piece", "black checker piece", "white checker piece"],
#         "MAH": ["mahjong tile", "bamboo mahjong tile", "character mahjong tile"],
#         "LEG": ["lego piece", "green lego piece", "light pink lego piece"],
#         "CHS": ["chess piece", "black chess piece", "white chess piece"],
#         "PZP": ["puzzle piece", "edge puzzle piece", "center puzzle piece"],
#         "PUZ": ["puzzle piece", "edge puzzle piece", "center puzzle piece"],
#         "PKC": ["poker chip", "blue poker chip", "white poker chip"],
#         "PLC": ["playing card", "red playing card", "black playing card"],
#         "MAR": ["marble", "big marble", "small marble"],
#         "DIC": ["dice", "green dice", "white dice"],
#         "CSC": ["chinese slim card", "chinese slim card without red marks", "chinese slim card with red marks"]
#     },
#     "HOU": {
#         "TPK": ["toothpick", "straight plastic toothpick", "dental floss"],
#         "CTB": ["cotton bud", "wooden cotton bud", "plastic cotton bud"],
#         "PIL": ["pill", "white pill", "yellow pill"],
#         "BAT": ["battery", "small AAA battery", "big AA battery"],
#         "HCP": ["hair clipper", "black hair clipper", "brown hair clipper"],
#         "MNY": ["money bill", "1000 vietnamese dong bill", "5000 vietnamese dong bill"],
#         "COI": ["coin", "5 Australian cents coin", "10 Australian cents coin"],
#         "BOT": ["bottle cap", "beer bottle cap", "plastic bottle cap"],
#         "BBT": ["button", "button with 4 holes", "button with 2 holes"],
#         "ULT": ["plastic utensil", "plastic spoon", "plastic fork"]
#     },
#     "OFF": {
#         "PPN": ["push pin", "normal push pin", "round push pin"],
#         "HST": ["heart sticker", "big heart sticker", "small heart sticker"],
#         "CRS": ["craft stick", "red or orange craft stick", "blue or purple craft stick"],
#         "RUB": ["rubber band", "yellow rubber band", "blue rubber band"],
#         "STN": ["sticky note", "dark green sticky note", "light green sticky note"],
#         "PPC": ["paper clip", "colored paper clip", "silver paper clip"],
#         "PEN": ["pen", "pen with cap", "pen without cap"],
#         "PNC": ["pencil"],
#         "RHS": ["rhinestone", "round rhinestone", "star rhinestone"],
#         "ZPT": ["zip tie", "short zip tie", "long zip tie"],
#         "SFP": ["safety pin", "big safety pin", "small safety pin"],
#         "LPP": ["lapel pin"],
#         "WWO": ["wall wire organizer"]
#     },
#     "OTR": {
#         "SCR": ["screw", "long silver concrete screw", "short bronze screw"],
#         "BOL": ["bolt", "hex head bolt", "mushroom head bolt"],
#         "NUT": ["nut", "hex nut", "square nut"],
#         "WAS": ["washer", "metal washer", "nylon washer"],
#         "BUT": ["button", "Beige", "Clear"],
#         "NAI": ["nail", "common nail", "concrete nail"],
#         "BEA": ["bead", "Blue and purple", "Orange and pink"],
#         "IKC": ["ikea clip", "green ikea clip", "red ikea clip"],
#         "IKE": ["ikea clip", "green ikea clip", "red ikea clip"],
#         "PEG": ["peg", "grey peg", "white peg"],
#         "STO": ["stone", "red stone", "yellowstone"]
#     }
# }


def get_correct_object_description(super_category, object_code):
    """
    Get the correct object description based on super category and object code.
    
    Args:
        super_category: Super category (FOO, FUN, HOU, OFF, OTR)
        object_code: Object code (e.g., PAS, RIC, etc.)
        
    Returns:
        str: First (main) description for the object, or None if not found
    """
    if super_category in OBJECT_DESCRIPTIONS:
        if object_code in OBJECT_DESCRIPTIONS[super_category]:
            return OBJECT_DESCRIPTIONS[super_category][object_code][0]
    return None


def get_object_description_by_code_with_suffix(super_category, object_code_with_suffix):
    """
    Get the correct object description based on super category and object code with suffix.
    
    Args:
        super_category: Super category (FOO, FUN, HOU, OFF, OTR)
        object_code_with_suffix: Object code with suffix (e.g., PEG1, PEG2, COI1, COI2)
        
    Returns:
        str: Specific description for the object code with suffix, or None if not found
    """
    if super_category not in OBJECT_DESCRIPTIONS:
        return None
    
    # Extract base code and suffix
    # Handle cases like PEG1, PEG2, COI1, COI2, etc.
    import re
    match = re.match(r'^([A-Z]+)(\d+)$', object_code_with_suffix)
    if not match:
        # If no suffix, try to get the base description
        return get_correct_object_description(super_category, object_code_with_suffix)
    
    base_code = match.group(1)
    suffix_num = int(match.group(2))
    
    if base_code in OBJECT_DESCRIPTIONS[super_category]:
        descriptions = OBJECT_DESCRIPTIONS[super_category][base_code]
        # suffix_num corresponds directly to the index in the descriptions list
        # So PEG1 -> index 1, PEG2 -> index 2, etc.
        if 1 <= suffix_num < len(descriptions):
            return descriptions[suffix_num]
        else:
            # If suffix is out of range, return the first description
            print(f"Warning: Suffix {suffix_num} out of range for {base_code}, using first description")
            return descriptions[0]
    
    return None


def parse_filename_components(filename):
    """
    Parse filename to extract all components including object codes.
    
    Example filename: 5-cents-coin_10-cents-coin_INTRA_HOU_COI1_COI2_00040_00010_1_00258.jpg
    Format: {obj1_name}_{obj2_name}_{test_type}_{super_category}_{pos_code}_{neg_code}_{pos_count}_{neg_count}_{id1}_{id2}.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        dict: Dictionary with parsed components
    """
    # Remove extension
    name_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    components = {
        'obj1_name': None,
        'obj2_name': None,
        'test_type': None,
        'super_category': None,
        'pos_code': None,
        'neg_code': None,
        'pos_count': None,
        'neg_count': None,
        'id1': None,
        'id2': None
    }
    
    if len(parts) >= 10:
        try:
            components['obj1_name'] = parts[0]
            components['obj2_name'] = parts[1]
            components['test_type'] = parts[2].upper()
            components['super_category'] = parts[3].upper()
            components['pos_code'] = parts[4]
            components['neg_code'] = parts[5]
            components['pos_count'] = int(parts[6])
            components['neg_count'] = int(parts[7])
            components['id1'] = parts[8]
            components['id2'] = parts[9]
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not fully parse filename components: {filename}, error: {e}")
    
    return components


def correct_object_name(original_name, super_category, object_code):
    """
    Correct object name based on the standard descriptions.
    
    Args:
        original_name: Original object name from filename
        super_category: Super category
        object_code: Object code (may include suffix like PEG1, PEG2)
        
    Returns:
        str: Corrected object name, or original if no correction found
    """
    correct_desc = get_object_description_by_code_with_suffix(super_category, object_code)
    if correct_desc:
        # Convert to filename format (replace spaces with hyphens)
        corrected_name = correct_desc.replace(' ', '-')
        if original_name != corrected_name:
            print(f"Correcting object name: '{original_name}' -> '{corrected_name}' (category: {super_category}, code: {object_code})")
            return corrected_name
    return original_name


def generate_corrected_filename(filename):
    """
    Generate a corrected filename with proper object descriptions.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Corrected filename
    """
    components = parse_filename_components(filename)
    
    if not all(components[key] is not None for key in ['super_category', 'pos_code', 'neg_code']):
        print(f"Warning: Could not parse all required components from {filename}, returning original")
        return filename
    
    # Get corrected object names
    corrected_obj1 = correct_object_name(
        components['obj1_name'], 
        components['super_category'], 
        components['pos_code']
    )
    corrected_obj2 = correct_object_name(
        components['obj2_name'], 
        components['super_category'], 
        components['neg_code']
    )
    
    # Reconstruct filename if any corrections were made
    if corrected_obj1 != components['obj1_name'] or corrected_obj2 != components['obj2_name']:
        extension = '.jpg'  # Default extension
        if '.' in filename:
            extension = '.' + filename.split('.')[-1]
        
        corrected_filename = f"{corrected_obj1}_{corrected_obj2}_{components['test_type']}_{components['super_category']}_{components['pos_code']}_{components['neg_code']}_{components['pos_count']:05d}_{components['neg_count']:05d}_{components['id1']}_{components['id2']}{extension}"
        
        print(f"Generated corrected filename: '{filename}' -> '{corrected_filename}'")
        return corrected_filename
    
    return filename


def parse_filename_for_counts(filename):
    """
    Parse filename to extract positive and negative object counts.
    
    Example filename: 5-cents-coin_10-cents-coin_INTRA_HOU_COI1_COI2_00040_00010_1_00258.jpg
    Format: {obj1_name}_{obj2_name}_{test_type}_{super_category}_{pos_code}_{neg_code}_{pos_count}_{neg_count}_{id1}_{id2}.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        tuple: (pos_count, neg_count)
    """
    # Remove extension
    name_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    # The pattern should be: obj1_obj2_TEST_CATEGORY_CODE1_CODE2_COUNT1_COUNT2_ID1_ID2
    # We need to find the counts, which should be the 6th and 7th parts from the end
    if len(parts) >= 8:
        try:
            # Get the last several parts and try to find the count pattern
            # Expected format: ..._CODE1_CODE2_COUNT1_COUNT2_ID1_ID2
            pos_count = int(parts[-4])  # 4th from end should be positive count
            neg_count = int(parts[-3])  # 3rd from end should be negative count
            return pos_count, neg_count
        except (ValueError, IndexError):
            pass
    
    # Fallback: try to find patterns with leading zeros
    pattern = r'_(\d{5})_(\d{5})_\d+_\d+\.?'
    match = re.search(pattern, filename)
    if match:
        pos_count = int(match.group(1))
        neg_count = int(match.group(2))
        return pos_count, neg_count
    
    # Another fallback pattern
    pattern = r'_(\d{5})_(\d{5})_'
    matches = re.findall(pattern, filename)
    if matches:
        pos_count = int(matches[0][0])
        neg_count = int(matches[0][1])
        return pos_count, neg_count
    
    print(f"Warning: Could not parse counts from filename: {filename}")
    return 0, 0


def parse_filename_for_prompts(filename):
    """
    Parse filename to extract positive and negative object prompts with correction.
    
    Example filename: 5-cents-coin_10-cents-coin_INTRA_HOU_COI1_COI2_00040_00010_1_00258.jpg
    Format: {obj1_name}_{obj2_name}_{test_type}_{super_category}_{pos_code}_{neg_code}_{pos_count}_{neg_count}_{id1}_{id2}.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        tuple: (positive_prompt, negative_prompt)
    """
    # Parse filename components to get correct descriptions
    components = parse_filename_components(filename)
    
    if components['super_category'] and components['pos_code'] and components['neg_code']:
        # Get correct descriptions from the dictionary using the new function that handles suffixes
        positive_prompt = get_object_description_by_code_with_suffix(components['super_category'], components['pos_code'])
        negative_prompt = get_object_description_by_code_with_suffix(components['super_category'], components['neg_code'])
        
        # If correct descriptions found, use them
        if positive_prompt and negative_prompt:
            return positive_prompt, negative_prompt
    
    # Fallback to original parsing method
    name_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    parts = name_without_ext.split('_')
    
    if len(parts) >= 2:
        obj1_name = parts[0]
        obj2_name = parts[1]
        
        # Convert to prompts by replacing - with spaces
        positive_prompt = obj1_name.replace('-', ' ')
        negative_prompt = obj2_name.replace('-', ' ')
        
        return positive_prompt, negative_prompt
    
    print(f"Warning: Could not parse prompts from filename: {filename}")
    return "", ""


def parse_filename_for_test_type(filename):
    """
    Parse filename to extract test type (INTER or INTRA).
    
    Example filename: 5-cents-coin_10-cents-coin_INTRA_HOU_COI1_COI2_00040_00010_1_00258.jpg
    Format: {obj1_name}_{obj2_name}_{test_type}_{super_category}_{pos_code}_{neg_code}_{pos_count}_{neg_count}_{id1}_{id2}.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        str: Test type ('INTER', 'INTRA', or 'UNKNOWN')
    """
    # Remove extension
    name_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    # The test type should be the 3rd part (index 2)
    if len(parts) >= 3:
        test_type = parts[2].upper()
        if test_type in ['INTER', 'INTRA']:
            return test_type
    
    # Fallback: search for INTER or INTRA anywhere in the filename
    filename_upper = filename.upper()
    if 'INTER' in filename_upper:
        return 'INTER'
    elif 'INTRA' in filename_upper:
        return 'INTRA'
    
    print(f"Warning: Could not determine test type from filename: {filename}")
    return 'UNKNOWN'


def parse_filename_for_super_category(filename):
    """
    Parse filename to extract super category.
    
    Example filename: 5-cents-coin_10-cents-coin_INTRA_HOU_COI1_COI2_00040_00010_1_00258.jpg
    Format: {obj1_name}_{obj2_name}_{test_type}_{super_category}_{pos_code}_{neg_code}_{pos_count}_{neg_count}_{id1}_{id2}.jpg
    
    Args:
        filename: Image filename
        
    Returns:
        str: Super category ('HOU', 'OTR', 'FOO', etc., or 'UNKNOWN')
    """
    # Remove extension
    name_without_ext = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    # The super category should be the 4th part (index 3)
    if len(parts) >= 4:
        super_category = parts[3].upper()
        # Common super categories (can be extended as needed)
        known_categories = ['HOU', 'OTR', 'FOO', 'PLA', 'ANI', 'VEH', 'CLO', 'SPO', 'TOY', 'BOO']
        if super_category in known_categories or len(super_category) == 3:
            return super_category
    
    print(f"Warning: Could not determine super category from filename: {filename}")
    return 'UNKNOWN'


def filter_annotations_by_super_category(cvat_data, super_category):
    """
    Filter annotations to include only images of a specific super category.
    
    Args:
        cvat_data: Dictionary with CVAT annotations
        super_category: Super category to filter by (e.g., 'HOU', 'OTR', 'FOO')
        
    Returns:
        Dictionary with filtered CVAT annotations
    """
    filtered_data = {}
    
    for image_name, annotations in cvat_data.items():
        image_super_category = parse_filename_for_super_category(image_name)
        if image_super_category == super_category:
            filtered_data[image_name] = annotations
    
    return filtered_data


def get_unique_super_categories(cvat_data):
    """
    Get all unique super categories from the dataset.
    
    Args:
        cvat_data: Dictionary with CVAT annotations
        
    Returns:
        List of unique super categories found in the dataset
    """
    categories = set()
    
    for image_name in cvat_data.keys():
        category = parse_filename_for_super_category(image_name)
        if category != 'UNKNOWN':
            categories.add(category)
    
    return sorted(list(categories))


def filter_annotations_by_test_type(cvat_data, test_type):
    """
    Filter annotations to include only images of a specific test type.
    
    Args:
        cvat_data: Dictionary with CVAT annotations
        test_type: Test type to filter by ('INTER' or 'INTRA')
        
    Returns:
        Dictionary with filtered CVAT annotations
    """
    filtered_data = {}
    
    for image_name, annotations in cvat_data.items():
        image_test_type = parse_filename_for_test_type(image_name)
        if image_test_type == test_type:
            filtered_data[image_name] = annotations
    
    return filtered_data


def generate_random_points(count, image_bounds=None):
    """
    Generate random points within image bounds.
    
    Args:
        count: Number of points to generate
        image_bounds: Optional tuple (width, height), defaults to (1024, 1024)
        
    Returns:
        List of [x, y] coordinate pairs
    """
    if image_bounds is None:
        width, height = 1024, 1024
    else:
        width, height = image_bounds
    
    points = []
    for _ in range(count):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        points.append([x, y])
    
    return points


def infer_image_bounds_from_annotations(annotations):
    """
    Infer reasonable image bounds from existing bounding boxes.
    
    Args:
        annotations: Annotation data with pos/neg boxes
        
    Returns:
        tuple: (width, height) estimated bounds
    """
    all_coords = []
    
    # Collect all coordinates from positive and negative boxes
    for boxes_key in ['pos', 'neg']:
        if boxes_key in annotations:
            for bbox in annotations[boxes_key]:
                all_coords.extend([bbox[0], bbox[2]])  # x1, x2
                all_coords.extend([bbox[1], bbox[3]])  # y1, y2
    
    if all_coords:
        max_coord = max(all_coords)
        # Add some padding and round up to reasonable bounds
        if max_coord <= 512:
            return 512, 512
        elif max_coord <= 1024:
            return 1024, 1024
        else:
            return int(max_coord * 1.2), int(max_coord * 1.2)
    
    # Default fallback
    return 1024, 1024


def convert_bbox_to_corners(bbox):
    """
    Convert bounding box from [x1, y1, x2, y2] format to corner coordinates format.
    
    Args:
        bbox: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
        
    Returns:
        List of corner coordinates: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    """
    x1, y1, x2, y2 = bbox
    return [
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2]   # bottom-left
    ]


def convert_cvat_to_fsc147(cvat_data, mapping_strategy='compact'):
    """
    Convert CVAT annotation data to FSC147 annotation format with compact filenames.
    
    Args:
        cvat_data: Dictionary with CVAT annotations
        mapping_strategy: 'compact' for structured names, 'simple' for img_XXXXXX.jpg
        
    Returns:
        tuple: (fsc147_data, filename_mapping, metadata_mapping)
    """
    # Create filename mapping
    filename_mapping, reverse_mapping, metadata_mapping = create_filename_mapping(cvat_data, mapping_strategy)
    
    fsc147_data = {}
    
    for original_filename, annotations in cvat_data.items():
        # Get the new compact filename
        compact_filename = filename_mapping[original_filename]
        
        # Get metadata for this image
        metadata = metadata_mapping[compact_filename]
        
        # Use metadata instead of parsing filename again
        pos_count = metadata['positive_count']
        neg_count = metadata['negative_count']
        positive_prompt = metadata['positive_prompt']
        negative_prompt = metadata['negative_prompt']
        
        # Infer image bounds from existing boxes
        image_bounds = infer_image_bounds_from_annotations(annotations)
        
        # Generate random points based on counts
        points = generate_random_points(pos_count, image_bounds)
        negative_points = generate_random_points(neg_count, image_bounds)
        
        # Convert positive examples
        box_examples_coordinates = []
        if 'pos' in annotations:
            for bbox in annotations['pos']:
                box_examples_coordinates.append(convert_bbox_to_corners(bbox))
        
        # Convert negative examples
        negative_box_exemples_coordinates = []
        if 'neg' in annotations:
            for bbox in annotations['neg']:
                negative_box_exemples_coordinates.append(convert_bbox_to_corners(bbox))
        
        # Create FSC147 entry using compact filename as key
        fsc147_entry = {
            "points": points,
            "negative_points": negative_points,
            "box_examples_coordinates": box_examples_coordinates,
            "negative_box_exemples_coordinates": negative_box_exemples_coordinates,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt
        }
        
        fsc147_data[compact_filename] = fsc147_entry
        
        # Print info for verification
        print(f"Processed {compact_filename}: {pos_count} pos points, {neg_count} neg points")
        print(f"  Original: {original_filename}")
        print(f"  Positive prompt: '{positive_prompt}', Negative prompt: '{negative_prompt}'")
    
    return fsc147_data, filename_mapping, metadata_mapping


def create_train_test_val_split(image_names):
    """
    Create train/test/val split with all images in test set as requested.
    
    Args:
        image_names: List of image names (can be corrected filenames)
        
    Returns:
        Dictionary with train/test/val splits
    """
    return {
        "test": sorted(image_names)
    }


def generate_compact_filename(filename, image_counter=None):
    """
    Generate a compact filename using structured IDs instead of long descriptions.
    
    Format: {category}_{test_type}_{pos_code}_{neg_code}_{pos_count:03d}_{neg_count:03d}_{hash_short}.jpg
    Example: HOU_INTRA_COI1_COI2_040_010_a1b2c3.jpg
    
    Args:
        filename: Original filename
        image_counter: Optional counter for unique IDs
        
    Returns:
        str: Compact filename
    """
    components = parse_filename_components(filename)
    
    if not all(components[key] is not None for key in ['super_category', 'test_type', 'pos_code', 'neg_code']):
        print(f"Warning: Could not parse all required components from {filename}, returning original")
        return filename
    
    # Create a hash from the original filename for uniqueness
    hash_obj = hashlib.md5(filename.encode())
    hash_short = hash_obj.hexdigest()[:6]  # Use first 6 characters
    
    # Get extension
    extension = '.jpg'
    if '.' in filename:
        extension = '.' + filename.split('.')[-1]
    
    # Generate compact filename
    compact_filename = f"{components['super_category']}_{components['test_type']}_{components['pos_code']}_{components['neg_code']}_{components['pos_count']:03d}_{components['neg_count']:03d}_{hash_short}{extension}"
    
    return compact_filename


def generate_simple_id_filename(filename, image_counter):
    """
    Generate a simple ID-based filename.
    
    Format: img_{counter:06d}.jpg
    Example: img_000001.jpg, img_000002.jpg
    
    Args:
        filename: Original filename
        image_counter: Counter for unique IDs
        
    Returns:
        str: Simple ID-based filename
    """
    # Get extension
    extension = '.jpg'
    if '.' in filename:
        extension = '.' + filename.split('.')[-1]
    
    return f"img_{image_counter:06d}{extension}"


def create_filename_mapping(cvat_data, mapping_strategy='compact'):
    """
    Create a mapping from original filenames to new compact filenames.
    
    Args:
        cvat_data: Dictionary with CVAT annotations
        mapping_strategy: 'compact' for structured compact names, 'simple' for img_XXXXXX.jpg
        
    Returns:
        tuple: (filename_mapping, reverse_mapping, metadata_mapping)
               - filename_mapping: original -> new filename
               - reverse_mapping: new -> original filename  
               - metadata_mapping: new filename -> metadata dict
    """
    filename_mapping = {}
    reverse_mapping = {}
    metadata_mapping = {}
    image_counter = 1
    
    for original_filename in sorted(cvat_data.keys()):
        if mapping_strategy == 'simple':
            new_filename = generate_simple_id_filename(original_filename, image_counter)
        else:  # compact
            new_filename = generate_compact_filename(original_filename, image_counter)
        
        # Ensure uniqueness
        while new_filename in reverse_mapping:
            image_counter += 1
            if mapping_strategy == 'simple':
                new_filename = generate_simple_id_filename(original_filename, image_counter)
            else:
                new_filename = generate_compact_filename(original_filename, image_counter)
        
        filename_mapping[original_filename] = new_filename
        reverse_mapping[new_filename] = original_filename
        
        # Store metadata for the new filename
        components = parse_filename_components(original_filename)
        pos_count, neg_count = parse_filename_for_counts(original_filename)
        positive_prompt, negative_prompt = parse_filename_for_prompts(original_filename)
        test_type = parse_filename_for_test_type(original_filename)
        super_category = parse_filename_for_super_category(original_filename)
        
        metadata_mapping[new_filename] = {
            'original_filename': original_filename,
            'super_category': super_category,
            'test_type': test_type,
            'positive_code': components.get('pos_code'),
            'negative_code': components.get('neg_code'),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'positive_prompt': positive_prompt,
            'negative_prompt': negative_prompt,
            'components': components
        }
        
        image_counter += 1
    
    return filename_mapping, reverse_mapping, metadata_mapping


def main():
    parser = argparse.ArgumentParser(description='Convert CVAT annotations to FSC147 format')
    parser.add_argument('input_file', help='Input CVAT annotation file (parsed_annotations.json)')
    parser.add_argument('--output_dir', default='.', help='Output directory for generated files')
    parser.add_argument('--annotation_file', default='annotation_FSC147_384.json', 
                       help='Output annotation file name')
    parser.add_argument('--split_file', default='Train_Test_Val_FSC_147.json',
                       help='Output train/test/val split file name')
    parser.add_argument('--inter_annotation_file', default='annotation_FSC147_384_inter.json',
                       help='Output annotation file name for inter objects only')
    parser.add_argument('--intra_annotation_file', default='annotation_FSC147_384_intra.json',
                       help='Output annotation file name for intra objects only')
    parser.add_argument('--inter_split_file', default='Train_Test_Val_FSC_147_inter.json',
                       help='Output train/test/val split file name for inter objects')
    parser.add_argument('--intra_split_file', default='Train_Test_Val_FSC_147_intra.json',
                       help='Output train/test/val split file name for intra objects')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible point generation')
    parser.add_argument('--create_super_category_files', action='store_true',
                       help='Create separate annotation files for each super category (HOU, OTR, FOO, etc.)')
    parser.add_argument('--filename_mapping_file', default='filename_mapping.json',
                       help='Output file for filename mapping (original -> compact)')
    parser.add_argument('--metadata_file', default='image_metadata.json',
                       help='Output file for image metadata (compact filename -> metadata)')
    parser.add_argument('--mapping_strategy', choices=['compact', 'simple'], default='compact',
                       help='Filename mapping strategy: "compact" for structured names, "simple" for img_XXXXXX.jpg')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load CVAT annotations
    print(f"Loading CVAT annotations from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        cvat_data = json.load(f)
    
    print(f"Found {len(cvat_data)} annotated images")
    
    # Convert to FSC147 annotation format
    print(f"Converting to FSC147 annotation format using '{args.mapping_strategy}' filename mapping...")
    fsc147_annotations, filename_mapping, metadata_mapping = convert_cvat_to_fsc147(cvat_data, args.mapping_strategy)
    
    # Create train/test/val split (all in test as requested)
    print("Creating train/test/val split...")
    # Use new compact filenames from the FSC147 annotations
    compact_image_names = list(fsc147_annotations.keys())
    train_test_val_split = create_train_test_val_split(compact_image_names)
    
    # Write main output files
    annotation_output_path = os.path.join(args.output_dir, args.annotation_file)
    split_output_path = os.path.join(args.output_dir, args.split_file)
    
    print(f"Writing annotation file to {annotation_output_path}...")
    with open(annotation_output_path, 'w') as f:
        json.dump(fsc147_annotations, f, indent=2)
    
    print(f"Writing train/test/val split to {split_output_path}...")
    with open(split_output_path, 'w') as f:
        json.dump(train_test_val_split, f, indent=2)
    
    # Write filename mapping
    mapping_output_path = os.path.join(args.output_dir, args.filename_mapping_file)
    print(f"Writing filename mapping to {mapping_output_path}...")
    with open(mapping_output_path, 'w') as f:
        json.dump(filename_mapping, f, indent=2)
    
    # Write metadata mapping
    metadata_output_path = os.path.join(args.output_dir, args.metadata_file)
    print(f"Writing image metadata to {metadata_output_path}...")
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata_mapping, f, indent=2)
    
    # Create and write INTER-only annotations
    print("Creating INTER-only annotations...")
    # Filter using metadata instead of parsing filenames
    inter_fsc147_data = {k: v for k, v in fsc147_annotations.items() 
                         if metadata_mapping[k]['test_type'] == 'INTER'}
    if inter_fsc147_data:
        inter_image_names = list(inter_fsc147_data.keys())
        inter_train_test_val_split = create_train_test_val_split(inter_image_names)
        
        inter_annotation_output_path = os.path.join(args.output_dir, args.inter_annotation_file)
        inter_split_output_path = os.path.join(args.output_dir, args.inter_split_file)
        
        print(f"Writing INTER annotation file to {inter_annotation_output_path}...")
        with open(inter_annotation_output_path, 'w') as f:
            json.dump(inter_fsc147_data, f, indent=2)
        
        print(f"Writing INTER train/test/val split to {inter_split_output_path}...")
        with open(inter_split_output_path, 'w') as f:
            json.dump(inter_train_test_val_split, f, indent=2)
    else:
        print("Warning: No INTER images found!")
    
    # Create and write INTRA-only annotations
    print("Creating INTRA-only annotations...")
    # Filter using metadata instead of parsing filenames
    intra_fsc147_data = {k: v for k, v in fsc147_annotations.items() 
                         if metadata_mapping[k]['test_type'] == 'INTRA'}
    if intra_fsc147_data:
        intra_image_names = list(intra_fsc147_data.keys())
        intra_train_test_val_split = create_train_test_val_split(intra_image_names)
        
        intra_annotation_output_path = os.path.join(args.output_dir, args.intra_annotation_file)
        intra_split_output_path = os.path.join(args.output_dir, args.intra_split_file)
        
        print(f"Writing INTRA annotation file to {intra_annotation_output_path}...")
        with open(intra_annotation_output_path, 'w') as f:
            json.dump(intra_fsc147_data, f, indent=2)
        
        print(f"Writing INTRA train/test/val split to {intra_split_output_path}...")
        with open(intra_split_output_path, 'w') as f:
            json.dump(intra_train_test_val_split, f, indent=2)
    else:
        print("Warning: No INTRA images found!")
    
    # Create and write super category specific annotations (if requested)
    super_category_data = {}
    if args.create_super_category_files:
        print("Creating super category specific annotations...")
        # Get unique categories from metadata
        unique_categories = set()
        for metadata in metadata_mapping.values():
            category = metadata['super_category']
            if category != 'UNKNOWN':
                unique_categories.add(category)
        unique_categories = sorted(list(unique_categories))
        
        print(f"Found super categories: {unique_categories}")
        
        for category in unique_categories:
            print(f"Processing super category: {category}")
            # Filter FSC147 annotations by super category using metadata
            category_fsc147_data = {k: v for k, v in fsc147_annotations.items() 
                                  if metadata_mapping[k]['super_category'] == category}
            
            if category_fsc147_data:
                category_image_names = list(category_fsc147_data.keys())
                category_train_test_val_split = create_train_test_val_split(category_image_names)
                
                # Create filenames for this super category
                category_annotation_file = f'annotation_FSC147_384_{category.lower()}.json'
                category_split_file = f'Train_Test_Val_FSC_147_{category.lower()}.json'
                
                category_annotation_output_path = os.path.join(args.output_dir, category_annotation_file)
                category_split_output_path = os.path.join(args.output_dir, category_split_file)
                
                # Write files
                print(f"Writing {category} annotation file to {category_annotation_output_path}...")
                with open(category_annotation_output_path, 'w') as f:
                    json.dump(category_fsc147_data, f, indent=2)
                
                print(f"Writing {category} train/test/val split to {category_split_output_path}...")
                with open(category_split_output_path, 'w') as f:
                    json.dump(category_train_test_val_split, f, indent=2)
                
                # Store data for summary
                super_category_data[category] = {
                    'annotation_path': category_annotation_output_path,
                    'split_path': category_split_output_path,
                    'count': len(category_fsc147_data)
                }
            else:
                print(f"Warning: No images found for super category {category}!")
    
    print("Conversion completed successfully!")
    print(f"Generated files:")
    print(f"  - {annotation_output_path}")
    print(f"  - {split_output_path}")
    print(f"  - {mapping_output_path}")
    print(f"  - {metadata_output_path}")
    if inter_fsc147_data:
        print(f"  - {inter_annotation_output_path}")
        print(f"  - {inter_split_output_path}")
    if intra_fsc147_data:
        print(f"  - {intra_annotation_output_path}")
        print(f"  - {intra_split_output_path}")
    
    # List super category files
    if super_category_data:
        print(f"  Super category files:")
        for category, data in super_category_data.items():
            print(f"    - {data['annotation_path']}")
            print(f"    - {data['split_path']}")
    
    # Print summary statistics
    total_pos_boxes = sum(len(data.get('pos', [])) for data in cvat_data.values())
    total_neg_boxes = sum(len(data.get('neg', [])) for data in cvat_data.values())
    total_pos_points = sum(len(data.get('points', [])) for data in fsc147_annotations.values())
    total_neg_points = sum(len(data.get('negative_points', [])) for data in fsc147_annotations.values())
    
    # Count images by test type using metadata
    inter_count = len(inter_fsc147_data) if 'inter_fsc147_data' in locals() else 0
    intra_count = len(intra_fsc147_data) if 'intra_fsc147_data' in locals() else 0
    unknown_count = len(fsc147_annotations) - inter_count - intra_count
    
    print(f"\nSummary:")
    print(f"  - Total images: {len(cvat_data)} (original) -> {len(fsc147_annotations)} (with compact filenames)")
    print(f"  - Filename mapping strategy: {args.mapping_strategy}")
    print(f"    - INTER images: {inter_count}")
    print(f"    - INTRA images: {intra_count}")
    if unknown_count > 0:
        print(f"    - Unknown type images: {unknown_count}")
    
    # Add super category breakdown
    if super_category_data:
        print(f"  - Super category breakdown:")
        for category, data in super_category_data.items():
            print(f"    - {category}: {data['count']} images")
    
    print(f"  - Positive boxes: {total_pos_boxes}")
    print(f"  - Negative boxes: {total_neg_boxes}")
    print(f"  - Positive points generated: {total_pos_points}")
    print(f"  - Negative points generated: {total_neg_points}")
    print(f"  - All images assigned to 'test' split")
    print(f"  - Random seed used: {args.seed}")
    print(f"  - Positive and negative prompts extracted from standardized object descriptions")
    if args.create_super_category_files:
        print(f"  - Super category specific files created")
    
    # Print filename mapping examples
    print(f"\nFilename mapping examples:")
    example_count = min(5, len(filename_mapping))
    for i, (original, compact) in enumerate(list(filename_mapping.items())[:example_count]):
        print(f"  '{original}' -> '{compact}'")
    if len(filename_mapping) > example_count:
        print(f"  ... and {len(filename_mapping) - example_count} more")
    
    print(f"\nComplete filename mapping saved to: {mapping_output_path}")
    print(f"Image metadata (including original filenames) saved to: {metadata_output_path}")


if __name__ == "__main__":
    main()