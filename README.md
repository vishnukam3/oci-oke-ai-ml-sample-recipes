# Corrino Recipe Documentation
## Deploy a Recipe
`POST /deploy`
### Request Body
| Parameter | Type | Required | Description
|-----------|------|----------|-------------
| recipe_id | string | Yes | One of the following: `llm_inference_nvidia`, `lora_finetune_nvidia`, or  `mlcommons_lora_finetune_nvidia`
| deployment_name | string | Yes | Any deployment name to identify the deployment details easily. Must be unique from other recipe deployments.
| recipe_mode | string | Yes | One of the following: `service` or `job`. Enter `service` for inference recipe deployments and `job` for fine-tuning recipe deployments.
| service_endpoint_domain | string | No | Required for inference recipe deployments. Inference endpoint will point to this domain.
| recipe_container_port | string | No | Required for inference recipe deployments. Inference endpoint will point to this port.
| recipe_node_shape | string | Yes | Enter the shape of the node that you want to deploy the recipe on to. Example: `BM.GPU.A10.4`
| recipe_node_pool_size | int | Yes | Number of nodes that you want to allocate for this recipe deployment. Ensure you have sufficient capacity. This feature is under development. Always enter 1.
| recipe_nvidia_gpu_count | int | Yes | Number of GPUs within the node that you want to deploy the recipe's artifacts on to. Must be greater than 0. Must be less than the total number of GPUs available in the node shape. For example, `VM.GPU.A10.2` has 2 GPUs, so this parameter cannot exceed 2 if the `recipe_node_shape` is `VM.GPU.A10.2`.
| recipe_replica_count | int | Yes | Number of replicas of the recipe container pods to create. This feature is under development. Always enter 1.
| recipe_ephemeral_storage_size | int | Yes | Ephemeral (will be deleted) storage in GB to add to node. If pulling large models from huggingface directly, set this value to be reasonably high. Cannot be higher than `boot_volume_size`.
| recipe_node_boot_volume_size_in_gbs | int | Yes | Size of boot volume in GB for image. Recommend entering 500.???
| recipe_shared_memory_volume_size_limit_in_mb | int | Yes | ???. Recommend entering 100.??
| input_object_storage | object | Yes | Name of bucket to mount at location “mount_location”. Mount size will be `volume_size_in_gbs`. Will copy all objects in bucket to mount location. Store your LLM model (and in the case of fine-tuning recipes, your input dataset as well) in this bucket. Example: `[{"bucket_name": "corrino_hf_oss_models", "mount_location": "/models", "volume_size_in_gbs": 500}]`
| output_object_storage | object | No | Required for fine-tuning deployments. Name of bucket to mount at location “mount_location”. Mount size will be “volume_size_in_gbs”. Will copy all items written here during program runtime to bucket on program completion. Example: `[{“bucket_name”: “output”,“mount_location”: “/output”,“volume_size_in_gbs”: 500}]`
| recipe_image_uri | string | Yes | Location of the recipe container image. Each recipe points to a specific container image. See the recipe.json examples below. Example: `iad.ocir.io/iduyx1qnmway/corrino-devops-repository:vllmv0.6.2`
| recipe_container_command_args | string | No | Container init arguments to pass. Each recipe has specific container arguments that it expects. See the Recipe Arguments section below for details. Example: `["--model","$(Model_Path)","--tensor-parallel-size","$(tensor_parallel_size)"]`
| recipe_container_env | string | No | Values of the recipe container init arguments. See the Recipe Arguments section below for details. Example: `[{"key": "tensor_parallel_size","value": "2"},{"key": "model_name","value": "NousResearch/Meta-Llama-3.1-8B-Instruct"},{"key": "Model_Path","value": "/models/NousResearch/Meta-Llama-3.1-8B-Instruct"}]`

### Recipe Container Arguments
#### LLM Inference using NVIDIA shapes and vLLM
This recipe deploys the vLLM container image. Follow the vLLM docs to pass the container arguments. See here: https://docs.vllm.ai/en/v0.5.5/serving/env_vars.html
#### MLCommons Llama-2 Quantized 70B LORA Fine-Tuning on A100
???
#### LORA Fine-Tune
| Argument | Example | Description
|----------|-------|-------------------
|Mlflow_Endpoint|http://mlflow.default.svc.cluster.local:5000|Internal routing to mlflow endpoint. Should not change.
|Mlflow_Exp_Name|Corrino_nvidia_recipe|Top level MLFlow experiment name
|Mlflow_Run_Name|Corrino_run|Lower level MLFlow run name inside experiment
|Hf_Token|hf_123456dfsalkj|Huggingface token used to authenticate for private models or datasets
|Download_Dataset_From_Hf|True or False|True if you want to download your dataset from huggingface. False if bringing your own from object storage
|Dataset_Name|tau/scrolls|Name of dataset. Only required if pulling from huggingface
|Dataset_Sub_Name|gov_report|If dataset has multiple sub-datasets like tau/scrolls, the name of the sub-dataset to use
|Dataset_Column_To_Use|None|Column of data-set to use for fine-tuning. Will try one of: [input, quote, instruct, conversations] if none is passed. Will error if can’t find column to use.
|Dataset_Path|/dataset|Path to local dataset, or path to dataset cache if downloading from hf
|Download_Model_From_Hf|True or False|True if you want to download your model from huggingface. False if bringing your own from object storage. Private models require hf_token set with proper permissions.
|Model_Name|meta-llama/Llama-3.2-1B-Instruct|Name of model. Only required if pulling from huggingface
|Model_Path|/models/|meta-llama/Llama-3.2-1B-Instruct	Path to local model, or path to model cache if downloading from hf
|Max_Model_Len|8192|Maximum positional embeddings of the model. Affects memory usage
|Resume_From_Checkpoint|True or False|Whether or not to resume from a previous checkpoint. In this case, model should be same base model used.
|Checkpoint_Path|/checkpoints/checkpoint-150|Path to mounted checkpoint, if resuming from checkpoint
|Lora_R|8|LoRA attention dimension
|Lora_Alpha|32|Alpha param for LoRA scaling
|Lora_Dropout|0.1|Dropout probability for LoRA layers
|Lora_Target_Modules|q_proj,up_proj,o_proj,k_proj,down_proj,gate_proj,v_proj|LoRA modules to use
|Bias|none|LoRA config bias
|Task_Type|CAUSAL_LM|LoRA config task type
|Per_Device_Train_Batch_Size|1|Batch size per GPU for training
|Gradient_Accumulation_Steps|1|Number of update steps to accumulate the gradients for before performing the backward / forward pass
|Warmup_Steps|2|Number of steps used for linear warmup from 0 to learning_rate
|Save_Steps|100|Write checkpoints every N steps
|Learning_Rate|0.0002|Initial learning rate for AdamW optimizer
|Fp16|True or False|Use fp16 data type
|Logging_Steps|1|Number of update steps between two logs
|Output_Dir|/outputs/Llama-3.2-1B-scrolls-tuned|Path to save model tuning output
|Optim|paged_adamw_8bit|Optimize to use
|Num_Train_Epochs|2|Total number of training epochs to perform
|Require_Persistent_Output_Dir|True or False|Validate that output directory is a mount location (this should be true for cloud runs wanting to write to “output_object_storage”)


### Recipe.json Examples
There are 3 recipes that we are providing out of the box. Following are example recipe.json snippets that you can use to deploy the recipes quickly for a test run.
|Recipe|Scenario|Sample JSON|
|----|----|----
|LLM Inference using NVIDIA shapes and vLLM|Deployment with default Llama-3.1-8B model using PAR|View sample JSON here [here](./vllm_inference_sample_recipe.json)
|MLCommons Llama-2 Quantized 70B LORA Fine-Tuning on A100|Default deployment with model and dataset ingested using PAR|View sample JSON here [here](./mlcommons_lora_finetune_nvidia_sample_recipe.json)
|LORA Fine-Tune Recipe|Open Access Model Open Access Dataset Download from Huggingface (no token required)|View sample JSON [here](./open_model_open_dataset_hf.backend.json)
|LORA Fine-Tune Recipe|Closed Access Model Open Access Dataset Download from Huggingface (Valid Auth Token Is Required!!)|View sample JSON [here](./closed_model_open_dataset_hf.backend.json)
|LORA Fine-Tune Recipe|Bucket Model Open Access Dataset Download from Huggingface (no token required)|View sample JSON [here](./bucket_model_open_dataset_hf.backend.json)
|LORA Fine-Tune Recipe|Get Model from Bucket in Another Region / Tenancy using Pre-Authenticated_Requests (PAR) Open Access Dataset Download from Huggingface (no token required)|View sample JSON [here](./bucket_par_open_dataset.backend.json)
|LORA Fine-Tune Recipe|Bucket Model Bucket Checkpoint Open Access Dataset Download from Huggingface (no token required)|View sample JSON [here](./bucket_checkpoint_bucket_model_open_dataset.backend.json)

## Undeploy a Recipe
`POST /undeploy`

## View available GPU Capacity in your region
`GET /oci_gpu_capacity/`

## View workspace details, including Prometheus, Grafana, and MLFlow URL
`GET /workspace/`

## View inference recipe deployment endpoint
`GET /workspace/`

## View deployment logs
`GET /deployment_logs/`

## Frequently Asked Questions

**Can I deploy custom models?**
Yes. Store your custom models and datasets in an Object Storage bucket. Point to that object storage bucket using the `input_obect_storage` bucket in the `/deploy` request body to deploy the recipe using your custom model or dataset.

**Can I create my own recipes?**
Yes, you must create a recipe container, move it to a container registry, and point to it using the `recipe_image_uri` field in the `/deploy` request body. 

**Can I orchestrate multiple models / recipes together?**
Yes

**I want to test this on larger GPUs – how can I do that?**
Please contact us and we can set it up for you.

**Where is the fine-tuned model saved?**
In an object storage bucket in the sandbox tenancy.

**Do you have a RAG recipe?** We have several other recipes that we have not exposed on the portal. If you would like any specific recipes that might better meet your needs, please contact us. 

**Is this built on top of OKE?** Yes. 

## Running into any issues?
Contact Vishnu Kammari at vishnu.kammari@oracle.com 