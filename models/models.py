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
    else:
        raise Exception("Can not find model!")

    print("model [%s] was created" % (model.model_names))
    return model
