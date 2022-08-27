// Import template file.
local template = import "template/bert-crf.libsonnet";
local plm_dir = "/nas-alinlp/linzhang.zx/models/";
// local plm_dir = "/home/data/embedding/";
local plm_name = "nezha-base";

local pseudo = std.extVar("TT_PSEUDO");

local data = {
    dir: "data/address/",
    reader: {
        type: "base",
        extend: if pseudo == '' then null else [pseudo],
        token_indexers: {
            bert: {
                type: "pretrained_transformer",
                model_name: plm_dir + "bert-base-chinese",
                tokenizer_kwargs: { do_lower_case: false }
            }
        }
    },
};

local model = {
    type: "plm_crf_tagger",
    plm: { model_name: plm_dir + plm_name },
    encoder: {
        type: "lstm",
        input_size: 768,
        hidden_size: 384,
        num_layers: 1,
        bidirectional: true
    },
    // encoder: { type: "pass_through", input_dim:768 }
};

local trainer = {
    // cuda_device: -1,
    num_epochs: 15,
    patience: 5,
    // grad_norm: std.parseJson(std.extVar("grad_norm")),
    optimizer: {
        type: "huggingface_adamw",
        lr: 0.001,
        // lr: std.parseJson(std.extVar("lr")),
        // eps: std.parseJson(std.extVar("eps")),
        weight_decay: 0.01,
        // correct_bias: std.parseJson(std.extVar("correct_bias")),
        parameter_groups: [
            [[".*transformer_model((?!(bias|LayerNorm)).)*$"], {"lr": 1e-5}],
            [[".*transformer_model.*(bias|LayerNorm)"], {"lr": 1e-5, "weight_decay": 0.0}],
        ]
    },
    // learning_rate_scheduler: {
    //     type: "linear_with_warmup",
    //     warmup_steps: std.parseInt(std.extVar("warmup_steps")),  // 100
    // },
};

template(data, model, trainer)
