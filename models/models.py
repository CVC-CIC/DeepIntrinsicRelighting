def create_model(opt):

    if opt.model_name == 'relighting_two_stage':
        from .two_stage_model import TwoStageModel
        model = TwoStageModel(opt)
    elif opt.model_name == 'relighting_two_stage_rs':
        from .two_stage_rs_model import TwoStageRSModel
        model = TwoStageRSModel(opt)
    elif opt.model_name == 'relighting_one_decoder':
        from .one_decoder_model import OneDecoderModel
        model = OneDecoderModel(opt)
    elif opt.model_name == 'drn':
        from .drn_model import DRNModel
        model = DRNModel(opt)
    elif opt.model_name == 'IAN':
        from .ian_model import IANModel
        model = IANModel(opt)
    elif opt.model_name == 'transformer':
        from .transformer_model import TransformerModel
        model = TransformerModel(opt)
    elif opt.model_name == 'vqgan':
        from .vqgan_model import VqganModel
        model = VqganModel(opt)
    elif opt.model_name == 'vqgan_intrinsic':
        from .vqgan_intrinsic_model import VqganIntrinsicModel
        model = VqganIntrinsicModel(opt)
    elif opt.model_name == 'transformer_intrinsic':
        from .transformer_intrinsic_model import TransformerIntrinsicModel
        model = TransformerIntrinsicModel(opt)
    else:
        raise Exception("Can not find model!")

    print("model [%s] was created" % (model.model_names))
    return model
