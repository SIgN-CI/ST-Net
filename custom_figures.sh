# patch_size=74
patch_size=296

gene=ALB
# gene=APOA1
# gene=MT2A

# patient=BC50027
patient=BC42334

bin/visualize.py /home/tester/bokleong/ST-Net6/output_inference/densenet121_296/top_250/BC42334_1.npz --gene ${gene} --figroot custom_fig/ --output_extension png --figure_spot_size 1500 --title_font_size 80
