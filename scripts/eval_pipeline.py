import asyncio
import os
import glob
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from backend.agents.intake_agent import run_intake
from backend.agents.identity_agent import run_identity
from backend.agents.gap_agent import run_gap

async def evaluate_board(board_name, image_paths):
    print(f"\n{'='*50}\nEvaluating Board: {board_name} ({len(image_paths)} images)\n{'='*50}")
    
    # 1. Intake
    print("Running Intake...")
    intake_res = await run_intake(image_paths)
    print(f"Clusters found: {len(intake_res.clusters)}")
    
    # 2. Identity
    print("Running Identity...")
    identity_res = await run_identity(intake_res)
    print(f"Primary World: {identity_res.primary_world}")
    print(f"Secondary World: {identity_res.secondary_world}")
    print(f"Visual Tension: {identity_res.visual_tension}")
    
    for w in identity_res.aesthetic_worlds:
        print(f"\n  - Cluster {w.cluster_id}: {w.name}")
        print(f"    Description: {w.description}")
        print(f"    Cultural Origin: {w.cultural_origin}")
        print(f"    Aspiration: {w.aspiration_reading}")
    
    # 3. Gap
    print("\nRunning Gap...")
    gap_res = await run_gap(intake_res, identity_res)
    for g in gap_res.all_gaps:
        print(f"\n  - GAP: {g.gap_name} ({g.severity})")
        print(f"    Aesthetic: {g.aesthetic}")
        print(f"    Requires: {g.what_it_requires}")
        print(f"    Common Miss: {g.common_miss}")
        print(f"    Tells: {', '.join(g.your_tell)}")
        print(f"    Action: {g.actionable_step}")
        
    return intake_res, identity_res, gap_res

async def main():
    base_dir = Path("data/pinterest_img")
    boards = ["minimal", "rock", "streetwear", "summer", "winter"]
    
    for board in boards:
        board_dir = base_dir / board
        if not board_dir.exists():
            print(f"Board {board} not found.")
            continue
            
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend(board_dir.rglob(ext))
            
        if not images:
            print(f"No images found for {board}.")
            continue
            
        # Sample 15 images to avoid excessive API calls if there are many
        images = images[:15]
        
        try:
            await evaluate_board(board, [os.path.abspath(p) for p in images])
        except Exception as e:
            print(f"Error evaluating {board}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
