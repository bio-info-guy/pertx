from transformers import PretrainedConfig

class TransformerModelConfig(PretrainedConfig):
    model_type = "sc_transformer"

    def __init__(
        self,
        ntoken=30000,
        d_model=512,
        nhead=8,
        d_hid=512,
        nlayers=12,
        nlayers_cls=3,
        n_cls=1,
        dropout=0.1,
        pad_token_id=0,     # Extracted from vocab[pad_token]
        mask_value=-1,
        pad_value=-2,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=None,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=None,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        gene_emb_style='vanilla',
        ecs_threshold=0.3,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
        expr_activation='linear',
        bin_num=10,
        bin_alpha=1.0,
        genept_path=None,
        cross_attn_decoder=False,
        decoder_layer=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ntoken = ntoken
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nlayers_cls = nlayers_cls
        self.n_cls = n_cls
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.use_batch_labels = use_batch_labels
        self.num_batch_labels = num_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.mvc_decoder_style = mvc_decoder_style
        self.gene_emb_style = gene_emb_style
        self.ecs_threshold = ecs_threshold
        self.explicit_zero_prob = explicit_zero_prob
        self.use_fast_transformer = use_fast_transformer
        self.pre_norm = pre_norm
        self.expr_activation = expr_activation
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        self.genept_path = genept_path
        self.cross_attn_decoder = cross_attn_decoder
        self.decoder_layer = decoder_layer

class PertTFGraphModelConfig(TransformerModelConfig):
    model_type = "pert_tf_graph_model"

    def __init__(
        self,
        n_pert=1000,
        nlayers_pert=3,
        n_ps=0,
        pert_graph_filename="pert_graph.pt", # Name of the tensor file
        pred_lochness_next=False,
        ps_decoder2_nlayer=3,
        pert_pad_id=None,
        pert_dim=None,
        sep_pert_mvc=False,
        pert_style='concat',
        gene_pert_shared=False,
        **kwargs
    ):
        # Pass base model args to the super constructor
        super().__init__(**kwargs)
        self.n_pert = n_pert
        self.nlayers_pert = nlayers_pert
        self.n_ps = n_ps
        self.pert_graph_filename = pert_graph_filename
        self.pred_lochness_next = pred_lochness_next
        self.ps_decoder2_nlayer = ps_decoder2_nlayer
        self.pert_pad_id = pert_pad_id
        self.pert_dim = pert_dim
        self.sep_pert_mvc = sep_pert_mvc
        self.pert_style = pert_style
        self.gene_pert_shared = gene_pert_shared