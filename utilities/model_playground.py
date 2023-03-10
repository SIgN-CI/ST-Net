import torch
import torchvision

def save_model_to_file(model,file):

    with open(file,"w") as f:
        try:
            f.write(f"model.state_dict().keys()=\n")
            for i, key in enumerate(model.state_dict().keys()):
                f.write(f"[{i:>3}]        {str(key)}\n")
            # f.write(f"{str(model.state_dict())}\n")
            f.write(f"{str(model.state_dict()['features.denseblock2.denselayer4.conv1.weight']) = }")

        except Exception as e:
            print(e)

            for i, key in enumerate(model):
                f.write(f"[{i:>3}] {str(key)}\n")

            # f.write(str(model))
            f.write(f"{str(model['features.denseblock2.denselayer4.conv1.weight']) = }")

if __name__ == "__main__":

    stnet_model = torchvision.models.__dict__["densenet121"](pretrained=True)
    save_model_to_file(stnet_model, "/home/tester/bokleong/ST-Net6/utilities/pretrained_weights.txt")

    # trained_model = torch.load("/home/tester/bokleong/ST-Net6/models/BC50027_epoch_100.pt")["model"]
    # save_model_to_file(trained_model, "/home/tester/bokleong/ST-Net6/utilities/BC50027_epoch_100_weights.txt")

    # loaded_model = torch.load("/home/tester/bokleong/ST-Net6/output_inference/densenet121_296/top_250/loaded_model.pt")
    # # print(loaded_model)
    # save_model_to_file(loaded_model, "/home/tester/bokleong/ST-Net6/utilities/loaded_model.txt")

    # print(trained_model["model"])
    # stnet_model.load_state_dict(trained_model)
    # # stnet_model.load_state_dict(torch.load("/home/tester/bokleong/ST-Net6/models/BC50027_epoch_100.pt").state_dict())
    # save_model_to_file(stnet_model,"/home/tester/bokleong/ST-Net6/utilities/after_load.txt")
