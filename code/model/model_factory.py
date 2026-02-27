# multi-path paradigm
# from model.clip_multi_path import CLIP_Multi_Path
# from model.coop_multi_path import COOP_Multi_Path
from model.troika import Troika
from model.action_adverb_model import ActionAdverbModel

def get_model(config, attributes=None, classes=None, offset=None,
              action_vocab=None, adverb_vocab=None, flow_dim=None, rgb_dim=None):
    """
    Factory function to create models.

    For Troika CZSL:
        Requires: attributes, classes, offset

    For ActionAdverbModel:
        Requires: action_vocab, adverb_vocab, flow_dim, rgb_dim
    """
    if config.model_name == 'troika':
        assert attributes is not None and classes is not None and offset is not None, \
            "Troika requires: attributes, classes, offset"
        model = Troika(config, attributes=attributes, classes=classes, offset=offset)
    elif config.model_name == 'action_adverb_model':
        assert action_vocab is not None and adverb_vocab is not None, \
            "ActionAdverbModel requires: action_vocab, adverb_vocab"
        assert flow_dim is not None and rgb_dim is not None, \
            "ActionAdverbModel requires: flow_dim, rgb_dim"
        model = ActionAdverbModel(config, action_vocab, adverb_vocab, flow_dim, rgb_dim)
    # elif config.model_name == 'clip_multi_path':
    #     model = CLIP_Multi_Path(config, attributes=attributes, classes=classes, offset=offset)
    # elif config.model_name == 'coop_multi_path':
    #     model = COOP_Multi_Path(config, attributes=attributes, classes=classes, offset=offset)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Model Name {:s}.".format(
                config.model_name
            )
        )

    return model