patch_size=74

gene=ALB
# gene=APOA1
# gene=MT2A

patient=BC50027

bin/visualize.py output/densenet121_${patch_size}/top_250/${patient}_cv.npz --gene ${gene} --figroot custom_fig/ --output_extension png
