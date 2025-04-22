import torch 
import torch.nn as nn 


def loadEMOModel(model_path: str, device: torch.device) -> nn.Module: 
    model = torch.jit.load(model_path).to(device)
    model.eval()
    return model

class VisualDiariz(nn.Module):
 
    def __init__(self, backbone: nn.Module, lstm: nn.Module):
        super().__init__()
        self.backbone = backbone #extracts per frame features
        self.lstm = lstm #extracts features for 10 frame indow 
        self.fc = nn.Linear(7, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("########################## EMONET X shape: ", x.shape)

        x = x.permute(0, 2, 1, 3, 4)  # 16 3 10 160 160 -> 16 10 3 160 160
        B, nF, C, H, W = x.shape #batch, number of consecutive frames (10), height (255), width (255)
        # x = x.view(B * nF, C, H, W) #need to process each frame individually
        x = x.reshape(B * nF, C, H, W)
        print("########################## EMONET X shape: ", x.shape)

        #extract conv features
        conv_feat = self.backbone.extract_features(x)  
        print("########################## EMONET CONVFEAT shape: ", conv_feat.shape)
        conv_dim = conv_feat.size(1)
        
        #extract ltsm features for a 10 frame window 
        lstm_input = conv_feat.view(B, nF, conv_dim)
        print("########################## EMONET LSTMINPUT shape: ", lstm_input.shape)
        lstm_out = self.lstm(lstm_input)
        print("########################## EMONET LSTMOUT shape: ", lstm_out.shape)
        
        
        rev_log = torch.logit(lstm_out, eps=1e-8) #ltsm outputs a (1,7) tensor of probabilities. apply torch.logit to center around 0 
        # out = self.fc(rev_log) #fully connected 7->1 logit tensor, apply sigmoid(out) for inference 
        
        return lstm_out 
    

def getModel(visD_weights_path: str, backbone_path: str, ltsm_path: str, device: torch.device) -> nn.Module: 
    
    backbone_model = loadEMOModel(backbone_path, device)
    ltsm_model = loadEMOModel(ltsm_path, device)
    visD_model = VisualDiariz(backbone_model, ltsm_model)
    
    state_dict = torch.load(visD_weights_path, map_location= device)
    visD_model.load_state_dict(state_dict)
    visD_model.eval()
    visD_model.to(device)
    
    return visD_model