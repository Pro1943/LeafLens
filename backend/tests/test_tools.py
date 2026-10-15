import numpy as np
import pytest

from leaflens.tools import TOOLS, clear_store, get_image, register_image


def test_tools_registry():
    clear_store()
    img = np.full((120, 120, 3), (35, 150, 45), dtype=np.uint8)
    image_id = register_image(img)

    assert "check_quality" in TOOLS
    assert "fix_exposure" in TOOLS
    assert "sharpen" in TOOLS
    assert "remove_glare" in TOOLS
    assert "crop_to_leaf" in TOOLS
    assert "segment_leaf" in TOOLS
    assert "measure_discoloration" in TOOLS
    assert "find_worst_patch" in TOOLS

    q_res = TOOLS["check_quality"]["fn"](image_id)
    assert "quality_report" in q_res

    exp_res = TOOLS["fix_exposure"]["fn"](image_id)
    assert "new_image_id" in exp_res
    assert get_image(exp_res["new_image_id"]) is not None

    sharp_res = TOOLS["sharpen"]["fn"](image_id, amount=1.2)
    assert "new_image_id" in sharp_res

    glare_res = TOOLS["remove_glare"]["fn"](image_id)
    assert "new_image_id" in glare_res
    assert "reconstructed_mask_id" in glare_res

    crop_res = TOOLS["crop_to_leaf"]["fn"](image_id, padding=5)
    assert "new_image_id" in crop_res

    seg_res = TOOLS["segment_leaf"]["fn"](image_id)
    assert "mask_image_id" in seg_res

    disc_res = TOOLS["measure_discoloration"]["fn"](
        image_id,
        seg_res["mask_image_id"],
        glare_res["reconstructed_mask_id"],
    )
    assert "stats" in disc_res
    assert "overlay_image_id" in disc_res
    assert "reconstructed_pct" in disc_res["stats"]

    patch_res = TOOLS["find_worst_patch"]["fn"](
        image_id,
        seg_res["mask_image_id"],
        glare_res["reconstructed_mask_id"],
    )
    assert "worst_patch" in patch_res
