import json
import os
import uuid

# Paths
HISTORY_PATH = "backend/rag/corpus/fashion_history.json"
EXECUTION_PATH = "backend/rag/corpus/aesthetic_execution.json"

def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

new_history = [
    # South Asian - Kurti
    {
        "id": generate_id("fh"),
        "aesthetic": "Kurti Aesthetics",
        "era": "Traditional to Modern",
        "chunk_type": "origin",
        "tags": ["kurti", "south asian", "everyday", "fusion", "silhouette"],
        "text": "The Kurti evolved from the traditional Kurta, a collarless shirt originating in the Indian subcontinent. While the Kurta was historically a looser, longer garment worn by both men and women, the modern Kurti adapted into a shorter, more fitted tunic for women. It bridges the gap between traditional ethnic wear and modern casual wear. The aesthetic relies heavily on the fabric choice\u2014block-printed cottons for everyday wear, Chanderi or silk for elevated looks. The silhouette's power comes from its adaptability, worn over churidars, palazzos, or even western denim, making it a cornerstone of contemporary South Asian daily fashion."
    },
    {
        "id": generate_id("fh"),
        "aesthetic": "Kurti Aesthetics",
        "era": "Traditional to Modern",
        "chunk_type": "visual_codes",
        "tags": ["kurti", "south asian", "proportion", "neckline", "slits"],
        "text": "The visual codes of the Kurti are defined by its neckline, sleeve length, and side slits (chaak). The side slits are not merely decorative but functional, allowing movement while dictating the drape. A high side slit paired with wide palazzos creates a fluid, A-line silhouette, whereas lower slits paired with cigarette pants create a sharp, structured column. Necklines (boat neck, mandarin collar, deep V) serve as the primary framing device for the face, often embellished with subtle embroidery (chikankari, zari) or striking piping."
    },
    # South Asian - Saree
    {
        "id": generate_id("fh"),
        "aesthetic": "Saree Styling",
        "era": "Ancient to Contemporary",
        "chunk_type": "origin",
        "tags": ["saree", "south asian", "drape", "handloom", "traditional"],
        "text": "The saree is one of the oldest unstitched garments in the world, originating in the Indian subcontinent. Its genius lies in its reliance on drape rather than tailoring. The traditional Nivi drape, standardized during the colonial era, is now the most globally recognized, but the garment holds regional fluidity (Bengali, Maharashtrian Nauvari, Gujarati). The aesthetic is deeply rooted in handloom heritage\u2014Kanjeevaram silks signaling southern luxury, Banarasi brocades for northern opulence, and crisp Bengal cottons for everyday elegance. The saree is an exercise in dynamic geometry, wrapping the body to create pleats (structure) and the pallu (fluidity)."
    },
    {
        "id": generate_id("fh"),
        "aesthetic": "Saree Styling",
        "era": "Ancient to Contemporary",
        "chunk_type": "visual_codes",
        "tags": ["saree", "blouse", "pallu", "pleats", "borders"],
        "text": "A saree's aesthetic impact is negotiated between three elements: the main drape, the pallu (the decorative end), and the blouse. The blouse anchors the fluid drape with structured tailoring\u2014ranging from traditional elbow-length sleeves to modern bralettes or high-neck halters. The visual tension often lives in the border (zari work or contrasting colors) which creates a sharp architectural line against the soft folds of the pleats. Modern styling often plays with these codes, belting the waist to control the drape or wearing the pallu structured and pleated rather than floating."
    },
    # South Asian - Traditional Fusion
    {
        "id": generate_id("fh"),
        "aesthetic": "Traditional Fusion",
        "era": "2000s to Present",
        "chunk_type": "origin",
        "tags": ["fusion", "south asian", "streetwear", "diaspora", "layering"],
        "text": "Traditional Fusion emerged primarily from the South Asian diaspora and modernizing metropolitan youth in the subcontinent. It is the conscious mixing of Western silhouettes with South Asian textiles, jewelry, or styling cues. This aesthetic treats traditional items not as occasion-wear, but as modular wardrobe pieces. It’s a rebellion against the rigid binary of 'western wear' and 'ethnic wear'. The cultural context is one of hyphenated identity\u2014claiming space in western environments by unapologetically integrating heritage codes, such as pairing a heavy Kundan choker with a graphic tee, or a vintage silk dupatta over an oversized blazer."
    },
    {
        "id": generate_id("fh"),
        "aesthetic": "Traditional Fusion",
        "era": "2000s to Present",
        "chunk_type": "visual_codes",
        "tags": ["fusion", "jhumkas", "bindi", "sneakers", "contrast"],
        "text": "The visual code of Traditional Fusion is intentional contrast. It relies on juxtaposing high-context cultural items (jhumkas, maang tikkas, bindis, or heavily embroidered jackets) with casual or utilitarian western staples (denim, sneakers, cargo pants). The success of the look depends on proportion and confidence; the traditional elements must feel integrated, not like a costume. A common execution is wearing a heavily worked lehenga skirt with a crisp white button-down shirt, or layering a Banarasi silk jacket over a minimalist slip dress."
    },
    # South Asian - Contemporary
    {
        "id": generate_id("fh"),
        "aesthetic": "Contemporary South Asian",
        "era": "2010s to Present",
        "chunk_type": "origin",
        "tags": ["contemporary", "couture", "modern", "sabyasachi", "minimalist ethnic"],
        "text": "Contemporary South Asian fashion represents a shift away from the heavy, stiff embellishments of late-20th-century bridal wear toward fluid luxury, tonal palettes, and revived heritage weaves. Designers like Sabyasachi Mukherjee reintroduced maximalist heritage but with editorial, cinematic styling, while labels like Raw Mango and Eka championed 'minimalist ethnic' by stripping away embroidery to focus entirely on the purity of the textile (chanderi, khadi). The aesthetic is characterized by a sophisticated understanding of scale\u2014large motifs, monochromatic dressing, and anti-fit silhouettes."
    },
    {
        "id": generate_id("fh"),
        "aesthetic": "Contemporary South Asian",
        "era": "2010s to Present",
        "chunk_type": "visual_codes",
        "tags": ["contemporary", "tonal", "anti-fit", "textile-focus", "layering"],
        "text": "The visual language of Contemporary South Asian fashion favors deep, rich tonal palettes (aubergine, mustard, ivory) over high-contrast blocking. It utilizes anti-fit silhouettes\u2014oversized kurtas, wide-leg shararas, and draped asymmetric tunics that float away from the body rather than constricting it. There is a strong emphasis on matte textures and quiet luxury within the ethnic space; replacing heavy zari (gold/silver thread) with subtle threadwork (resham), shadow work, or letting a pure handwoven textile speak for itself without any embellishment."
    }
]

new_execution = [
    # South Asian - Kurti
    {
        "id": generate_id("ae"),
        "aesthetic": "Kurti Aesthetics",
        "execution_marker": "Proportion and Slit Placement",
        "what_it_requires": "The side slits (chaak) must start at the exact right point on the hip to allow movement without compromising structure. When pairing with wide palazzos, a straight or A-line kurti with higher slits prevents the look from becoming a boxy block. When pairing with narrow pants, a tailored fit through the torso is essential.",
        "common_miss": "Wearing a kurti that is tight on the hips with low slits, causing the fabric to pull and bunch awkwardly across the stomach. Or pairing a very flared anarkali-style kurti with equally wide flared pants, drowning the figure.",
        "your_tell": ["slit height relative to hips", "fabric bunching at the waist", "balance between top volume and bottom volume"],
        "gap_type": "proportion",
        "severity": "critical",
        "actionable_step": "Ensure your side slits begin exactly at your hip bone for straight kurtis to allow clean drape over your lower half."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "Kurti Aesthetics",
        "execution_marker": "Fabric Context",
        "what_it_requires": "Everyday kurtis require breathable, matte fabrics like cotton or linen that hold a soft shape. Formal kurtis require silk, chanderi, or velvet that have inherent structure and sheen. The styling must match the fabric's context.",
        "common_miss": "Treating a synthetic, highly shiny polyester kurti as casual daywear, which looks contradictory. Alternatively, wearing a stiff, unwashed cotton kurti to a formal event where fluidity or richness is expected.",
        "your_tell": ["fabric sheen under daylight", "stiffness of the drape", "appropriateness for the occasion context"],
        "gap_type": "material",
        "severity": "moderate",
        "actionable_step": "Reserve synthetic or heavily embroidered fabrics for evening/formal contexts; stick to block-printed or solid cottons for daywear."
    },
    # South Asian - Saree
    {
        "id": generate_id("ae"),
        "aesthetic": "Saree Styling",
        "execution_marker": "Pleat Discipline and Drape Tension",
        "what_it_requires": "The front pleats must be sharply folded, even in width, and tucked cleanly so they fall straight down to the floor without billowing out. The pallu must have the correct tension\u2014neither choking the neck nor sagging sloppily at the armpit.",
        "common_miss": "Uneven, bulky pleats tucked too high, exposing the ankles and creating a bulky stomach. A pallu that slips constantly because it wasn't pinned correctly or the fabric tension across the chest is loose.",
        "your_tell": ["sharpness of the front pleats", "hemline straightness at the feet", "tension of the fabric across the bodice"],
        "gap_type": "styling",
        "severity": "critical",
        "actionable_step": "Iron your front pleats before tucking them in, and ensure the hem touches the floor while wearing your chosen heels."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "Saree Styling",
        "execution_marker": "Blouse Architecture",
        "what_it_requires": "A blouse that fits flawlessly. It should lie completely flat against the skin without gaping at the armholes or riding up at the back. The neckline should frame the collarbones purposefully.",
        "common_miss": "A poorly tailored blouse with gaping armholes, sagging shoulders, or bra straps showing unintentionally. A bad blouse ruins even the most expensive silk drape because it destroys the architectural foundation of the look.",
        "your_tell": ["armhole gaping", "shoulder seam sitting off the shoulder", "fabric riding up at the back"],
        "gap_type": "fit",
        "severity": "critical",
        "actionable_step": "Have your blouse tailored with proper darts and interfacing so it acts as a structured corset rather than a loose crop top."
    },
    # South Asian - Traditional Fusion
    {
        "id": generate_id("ae"),
        "aesthetic": "Traditional Fusion",
        "execution_marker": "Intentional Contrast",
        "what_it_requires": "A deliberate clash that feels curated. If wearing heavy traditional jewelry (like a Kundan choker), the accompanying western pieces must be inherently casual or minimalist (like a ribbed tank top or an oversized blazer) to create tension.",
        "common_miss": "Mixing 'middle-ground' pieces\u2014like a semi-formal western blouse with a semi-formal ethnic skirt. This doesn't look like fusion; it looks like you got dressed in the dark. The contrast must be extreme (high traditional vs. high casual/minimal).",
        "your_tell": ["level of formality contrast", "integration of jewelry", "color harmony between disparate pieces"],
        "gap_type": "styling",
        "severity": "moderate",
        "actionable_step": "Pair your most heavily embellished traditional accessory with your most basic, structured western basic (e.g., heavy earrings with a crisp white shirt)."
    },
    # South Asian - Contemporary
    {
        "id": generate_id("ae"),
        "aesthetic": "Contemporary South Asian",
        "execution_marker": "Tonal Layering and Anti-fit",
        "what_it_requires": "Layering garments of the same color family but different textures (e.g., an ivory silk kurta over ivory cotton palazzos with an ivory chanderi dupatta). The fit must be intentionally loose, skimming the body rather than hugging it tightly.",
        "common_miss": "Buying anti-fit clothing but sizing down because you're afraid of volume, resulting in a garment that just looks ill-fitting rather than intentionally oversized. Or breaking a sophisticated tonal look with jarring, brightly colored leggings.",
        "your_tell": ["volume of the silhouette", "color consistency", "drape of the fabric away from the body"],
        "gap_type": "proportion",
        "severity": "moderate",
        "actionable_step": "Embrace the volume; do not belt or tailor anti-fit garments to be tight, and keep the inner and outer layers in a strict monochromatic palette."
    },
    # Top 3 Deep Dive - Minimalism
    {
        "id": generate_id("ae"),
        "aesthetic": "90s Minimalism",
        "execution_marker": "Color Undertone Harmony",
        "what_it_requires": "When wearing neutrals (beiges, grays, whites), the undertones must match. Cool-toned grays with cool-toned whites; warm camel with warm ivory. The colors must look deliberate, not randomly assembled from a neutral pile.",
        "common_miss": "Mixing a cool, blue-toned stark white shirt with warm, yellow-toned cream trousers. The clash in undertones makes the outfit look accidental and muddy, breaking the pristine illusion of minimalism.",
        "your_tell": ["clashing neutral undertones", "stark white vs cream contrast", "muddiness in the palette"],
        "gap_type": "color",
        "severity": "moderate",
        "actionable_step": "Check your neutrals in natural daylight to ensure they share either a strictly warm or strictly cool undertone before pairing them."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "90s Minimalism",
        "execution_marker": "Seam and Hardware Restraint",
        "what_it_requires": "Garments where the construction is invisible. Hidden plackets covering buttons, welt pockets instead of patch pockets, blind hems on trousers, and zero unnecessary decorative stitching or exposed zippers.",
        "common_miss": "Buying a 'minimalist' coat that has contrasting plastic buttons, exposed shiny zippers, or chunky patch pockets. These details interrupt the clean lines and push the garment into casual/utilitarian territory.",
        "your_tell": ["visible hardware", "contrast stitching", "pocket style (patch vs welt)"],
        "gap_type": "construction",
        "severity": "critical",
        "actionable_step": "Examine the closures on your garments; replace cheap contrasting buttons with tonal ones, or seek out pieces with hidden plackets."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "90s Minimalism",
        "execution_marker": "The Crisp Hem Break",
        "what_it_requires": "Trousers must fall in a clean line from the hip and break exactly once (or not at all) at the shoe. A slight half-break or a clean crop that shows no sock.",
        "common_miss": "Trousers that pool around the ankles with multiple messy folds. This immediately ruins the structural discipline of minimalism and makes the wearer look sloppy and un-tailored.",
        "your_tell": ["fabric pooling at ankles", "multiple breaks in the trouser leg", "messy interaction with the shoe"],
        "gap_type": "fit",
        "severity": "critical",
        "actionable_step": "Take your trousers to a tailor with the exact shoes you intend to wear them with, and ask for a slight half-break or a clean no-break hem."
    },
    # Top 3 Deep Dive - Maximalism
    {
        "id": generate_id("ae"),
        "aesthetic": "Maximalism",
        "execution_marker": "Pattern Scale Discipline",
        "what_it_requires": "When clashing patterns, they must vary in scale. A large, bold floral paired with a tight, micro-check or a thin pinstripe. The eye needs a primary pattern and a secondary, background pattern to avoid visual chaos.",
        "common_miss": "Pairing two large, bold patterns of the exact same scale (e.g., a large polka dot with a large floral). The patterns fight for dominance, creating a dizzying, costume-like effect rather than curated maximalism.",
        "your_tell": ["scale of competing patterns", "visual resting space", "dominance of one pattern over the other"],
        "gap_type": "styling",
        "severity": "critical",
        "actionable_step": "Ensure one pattern acts as the 'anchor' (usually the larger one) and the second acts as a 'texture' (smaller, tighter pattern)."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "Maximalism",
        "execution_marker": "Color Threading",
        "what_it_requires": "Despite the chaos of layers, textures, and prints, there must be a common color thread that ties the elements together. A red accent in a floral skirt should be picked up by a red bag or red embroidery on a jacket.",
        "common_miss": "Throwing completely unrelated, highly saturated colors together without a unifying thread. This results in looking like you are wearing a disguise or fell into a thrift bin, rather than executing a deliberate editorial look.",
        "your_tell": ["absence of repeating colors", "randomness of palette", "lack of cohesion across layers"],
        "gap_type": "color",
        "severity": "moderate",
        "actionable_step": "Pick one specific shade from your busiest printed piece and repeat that exact shade somewhere else in the outfit via accessories or layering."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "Maximalism",
        "execution_marker": "Texture Layering",
        "what_it_requires": "Maximalism isn't just about prints; it's about tactile depth. Mixing a heavy, fuzzy knit with a slick vinyl trench, or a delicate lace with a stiff brocade. The contrast in textures provides the richness.",
        "common_miss": "Wearing many layers, but all of them are flat, matte cotton or polyester. Without textural variation, the look falls flat and feels heavy rather than rich and opulent.",
        "your_tell": ["variety of tactile surfaces", "light reflection on materials", "flatness vs depth"],
        "gap_type": "material",
        "severity": "moderate",
        "actionable_step": "Introduce at least one high-texture item (like faux fur, sequins, heavy velvet, or slick patent leather) to flat matte outfits."
    },
    # Top 3 Deep Dive - Y2K
    {
        "id": generate_id("ae"),
        "aesthetic": "Y2K Revival",
        "execution_marker": "The Exact Low-Rise Crop",
        "what_it_requires": "The low-rise must sit strictly below the belly button, resting on the hip bones. The top (baby tee, halter, or cardi) must crop precisely above the waistband, leaving a distinct 1-3 inch sliver of midriff. The tension lives in this specific gap.",
        "common_miss": "Wearing a 'mid-rise' that hits just under the belly button and pairing it with a top that overlaps the waistband. This completely misses the iconic Y2K silhouette and just looks like an ill-fitting 2010s outfit.",
        "your_tell": ["placement of the waistband relative to hip bones", "visibility of midriff", "top hem overlap"],
        "gap_type": "proportion",
        "severity": "critical",
        "actionable_step": "Ensure your jeans have a true 7-8 inch rise (or lower) and your top ends cleanly before the waistband begins."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "Y2K Revival",
        "execution_marker": "Chunky Footwear Proportions",
        "what_it_requires": "Footwear must have visual weight to anchor the flared or baggy pants. Platform boots, chunky skater shoes (like Osiris or DCs), or heavily padded sneakers. The pant hem must drape over the shoe, nearly swallowing it.",
        "common_miss": "Pairing wide-leg parachute pants or flared low-rise jeans with delicate, flat sandals, slim canvas sneakers, or pointed heels that peek out awkwardly. The bottom half loses its grounded, heavy anchor.",
        "your_tell": ["shoe volume", "pant hem interaction with shoe", "visual weight of the foot"],
        "gap_type": "proportion",
        "severity": "moderate",
        "actionable_step": "Swap slim profile shoes for platforms, lug soles, or heavily padded sneakers to balance wide Y2K pant hems."
    },
    {
        "id": generate_id("ae"),
        "aesthetic": "Y2K Revival",
        "execution_marker": "Hardware and Embellishment",
        "what_it_requires": "Visible, slightly tacky hardware. Rhinestone logos, oversized grommets on belts, chain details, metallic faux leather, and cargo pockets with exposed zippers. The aesthetic embraces conspicuous, plasticky, or metallic ornamentation.",
        "common_miss": "Trying to do 'minimalist' Y2K by just wearing plain low-rise jeans and a plain tee. Y2K requires a level of playful tackiness\u2014without the hardware or logos, it lacks the era's specific consumerist cultural code.",
        "your_tell": ["presence of rhinestones/studs", "visible branding", "belt styles (grommet, chain)"],
        "gap_type": "styling",
        "severity": "minor",
        "actionable_step": "Add a grommet belt, a rhinestone-embellished baby tee, or a baguette bag with heavy hardware to authentically signal the era."
    }
]

with open(HISTORY_PATH, "r", encoding="utf-8") as f:
    history_data = json.load(f)
history_data.extend(new_history)
with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history_data, f, indent=2)
print(f"Added {len(new_history)} chunks to {HISTORY_PATH}")

with open(EXECUTION_PATH, "r", encoding="utf-8") as f:
    execution_data = json.load(f)
execution_data.extend(new_execution)
with open(EXECUTION_PATH, "w", encoding="utf-8") as f:
    json.dump(execution_data, f, indent=2)
print(f"Added {len(new_execution)} chunks to {EXECUTION_PATH}")
