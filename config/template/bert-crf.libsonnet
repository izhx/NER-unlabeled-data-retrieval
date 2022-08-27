// template

local getattr (obj, name, default) = if std.objectHas(obj, name) then obj[name] else default;

local cuda = std.extVar("CUDA_VISIBLE_DEVICES");

function (data, model, trainer) {
    dataset_reader: data.reader + { label_namespace: "labels" },
    train_data_path: if "train" in data then data.train else data.dir + "/train.json",
    validation_data_path: if "dev" in data then data.dev else data.dir + "/dev.json",
    test_data_path: if "test" in data then data.test else data.dir + "/test.json",
    evaluate_on_test: if self.test_data_path == null then false else true,
    model: model + {
        initializer: {
            regexes: [
                ["tag_projection_layer.weight", {"type": "xavier_normal"}],
                ["encoder._module.weight_ih.*", {"type": "xavier_normal"}],
                ["encoder._module.weight_hh.*", {"type": "orthogonal"}]
            ]
        }
    },
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: std.parseInt(std.extVar("BATCH_SIZE")),
            sorting_keys: ["tokens"]
        }
    },
    random_seed: std.parseInt(std.extVar("RANDOM_SEED")),
    numpy_seed: self.random_seed,
    pytorch_seed: self.random_seed,
    trainer: {
        [if std.length(cuda) < 2 then "cuda_device"]: if std.length(cuda) > 0 then 0 else -1,
        patience: 3,
        validation_metric: "+f1-measure-overall",
        callbacks: [ { type: "distributed_test" } ],
        checkpointer: { keep_most_recent_by_count: 1 }
    } + trainer,
    vocabulary: { type: "from_files", directory: data.dir + "/vocabulary" },
    [if std.length(cuda) > 1 then "distributed"]: {
        cuda_devices: std.range(0, std.length(std.findSubstr(',', cuda)))
    }
}
