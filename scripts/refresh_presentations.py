"""Rebuild all offline flow presentation pages with the shared player."""
from pathlib import Path
import json,zipfile
import numpy as np
from animate_inverse_pair import export_player,render,PROJECT
from animate_comparison import load_comparison,load_data,sha256

def main():
    render()
    data_path=PROJECT/"data/raw/cylinder_nektar_wake.mat"
    xy,t,ids,pred,ref=load_comparison(PROJECT/"runs/forward_baseline",data_path,"auto")
    out=PROJECT/"output"
    export_player(xy,t,[ref,pred],out/"forward_comparison.html",["真值","正问题全窗 · Adam 50,000"],"正问题尾流对照")
    export_player(xy,t,[ref],out/"reference_animation.html",["数值模拟参考场"],"圆柱尾流 · 真值")
    for filename in ("forward_comparison","reference_animation"):
        meta=out/(filename+".source.json")
        value=json.loads(meta.read_text(encoding="utf-8-sig"))
        value.update(rendering="native rectangular grid, nearest pixels, no interpolation",
                     color_modes="shared reference range or full range; fixed across time",
                     times=t.tolist(),dataset_sha256=sha256(data_path))
        meta.write_text(json.dumps(value,indent=2),encoding="utf-8")
    package=out/"report_handoff_v1"
    with zipfile.ZipFile(package.with_suffix(".zip"),"w",zipfile.ZIP_DEFLATED) as z:
        for f in package.rglob("*"):
            if f.is_file():z.write(f,Path(package.name)/f.relative_to(package))
        for filename in ("forward_comparison.html", "forward_comparison.source.json",
                         "inverse_wake_comparison.html", "inverse_wake_comparison.source.json",
                         "forward_comparison.png"):
            z.write(out/filename, Path(package.name)/filename)
    print("Updated all pages and report package",flush=True)

if __name__=="__main__":
    main()
