// Import template file.
local template = import "template/bert-crf.libsonnet";
local plm_dir = "/nas-alinlp/linzhang.zx/models/";
local plm_name = "nezha-base";

local low = std.extVar("DATA_LOW");
local data = {
    dir: "data/ecommerce/",
    train: self.dir + "train.json.ret"  + if low == "" then "" else "." + low,
    dev: self.dir + "dev.json.ret",
    test: self.dir + "test.json.ret",
    reader: {
        type: "retrieval",
        max_retrieval: 10,
        max_length: 500,
        de_numbers: false,
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
    plm: {
        model_name: plm_dir + plm_name,
        max_length: data.reader.max_length + 100
    },
    encoder: {
        type: "lstm",
        input_size: 768,
        hidden_size: 384,
        num_layers: 1,
        bidirectional: true
    },
    embedding_dropout: 0.2,
    encoder_dropout: 0.2,
    // encoder: { type: "pass_through", input_dim:768 },
    top_k: 1
};


local trainer = {
    num_epochs: 20,
    patience: 5,
    // grad_norm: 1.0,
    validation_metric: "+f1-measure-overall",
    optimizer: {
        type: "huggingface_adamw",
        lr: 0.001,
        eps: 1e-8,
        weight_decay: 0.01,
        correct_bias: true,
        parameter_groups: [
            [[".*transformer_model((?!(bias|LayerNorm)).)*$"], {"lr": 1e-5}],
            [[".*transformer_model.*(bias|LayerNorm)"], {"lr": 1e-5, "weight_decay": 0}],
        ]
    },
    // learning_rate_scheduler: {
    //     type: "linear_with_warmup",
    //     warmup_steps: std.parseInt(std.extVar("warmup_steps")),  // 100
    // },
};

template(data, model, trainer)
