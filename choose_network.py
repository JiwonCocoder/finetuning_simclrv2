import torch.nn as nn
import pdb
import os
import torch
from torchsummary import summary
from resnet import get_resnet, name_to_params

def choose_network(args, net_from_name = True,
                   net = 'resnet50',
                   pretrained_from='scratch',
                   pretrained_model_dir='./pretrained_model',
                   ):
    # generating models from torchvision.models
    if net_from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))
        if net not in model_name_list: #토치비전에 모델 없으면 에러
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net}")
        else:
            # model = models.__dict__['wide_resnet50_2'](pretrained=True)
            # model.fc = nn.Linear(model.fc.in_features, 10)
            # return model

            net_model = models.__dict__[net]
            print(net_model.__name__+" is used")
            if pretrained_from == 'scratch':
                model = net_model(pretrained=False, num_classes=args.num_classes)
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            elif pretrained_from == 'ImageNet_supervised':
                model = net_model(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            elif "SimCLR" in pretrained_from:# CAUTION : 와이드레즈넷일때 따로 추가해줘야함.
                model = models.resnet50(pretrained=False, num_classes=args.num_classes)
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(name)
                model_name_from_path = os.path.join(pretrained_model_dir, net.lower())
                prefrained_from_path = pretrained_from.lower().replace("_", "/")
                checkpoint_dir = os.path.join(model_name_from_path, prefrained_from_path)

                checkpoint_file = os.path.join(checkpoint_dir,pretrained_from + ".pth")
                if os.path.isfile(checkpoint_file):
                    print("checkpoint_file_path:", checkpoint_file)
                    checkpoint = torch.load(checkpoint_file)
                    state_dict = checkpoint['state_dict']
                    # for k in list(state_dict.keys()):
                    #     if k.startswith('backbone.'):
                    #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    #             # remove prefix
                    #             state_dict[k[len("backbone."):]] = state_dict[k]
                    #             if k == 'backbone.conv1.weight':
                    #                 print("here")
                    #                 print(state_dict[k])
                    #     del state_dict[k]
                    for k in list(state_dict.keys()):
                        if k.startswith('fc'):
                            del state_dict[k]
                    log = model.load_state_dict(state_dict, strict=False)
                    print(log.missing_keys)
                    assert log.missing_keys == ['fc.weight', 'fc.bias']

                else:
                    checkpoint_file = os.path.join(checkpoint_dir, pretrained_from + '.ckpt')
                    assert os.path.isfile(checkpoint_file) == True
                    simclr = SimCLR.load_from_checkpoint(checkpoint_file, strict=False)
                    simclr_resnet50 = simclr.encoder
                    simclr_resnet50_wo_fc = torch.nn.Sequential(*(list(simclr_resnet50.children())[:-1]))
                    print(summary(simclr_resnet50.cuda(), (3, 128, 200)))
                    print(summary(model.cuda(), (3, 128, 200)))
                    for param_q, param_k in zip(simclr_resnet50_wo_fc.parameters(), model.parameters()):
                        param_k.data.copy_(param_q.detach().data)  # initialize


            elif "Simclrv2" in pretrained_from:
                print(pretrained_from)
                model_name_from_path = os.path.join(pretrained_model_dir, net.lower())
                print(model_name_from_path)
                pretrained_from_path = pretrained_from.lower().replace("_", "/")
                print(pretrained_from_path)
                checkpoint_dir = os.path.join(model_name_from_path, pretrained_from_path)
                checkpoint = os.listdir(checkpoint_dir)[-1]
                print(checkpoint)
                model, _ = get_resnet(*name_to_params(checkpoint))
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            return model
    else: # if net_from_name == false 인 경우
        # 여기도 if 문으로 프리트레인 조건 넣어줘야함. 그러나 net from name 을 항상 true 로 두면 그럴 필요 없음
        model = models.__dict__['wide_resnet50_2'](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        return model

