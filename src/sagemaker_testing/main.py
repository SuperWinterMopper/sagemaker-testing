import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel
def run_test():
    # WARNING: This snippet is not yet compatible with SageMaker version >= 3.0.0.
    # To use this snippet, install a compatible version:
    # pip install 'sagemaker<3.0.0'

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client('iam')
        role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

    # Hub Model configuration. https://huggingface.co/models
    hub = {
        'HF_MODEL_ID':'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'HF_TASK':'text-classification'
    }

    # create Hugging Face Model Classhow
    huggingface_model = HuggingFaceModel(
        transformers_version='4.51.3',
        pytorch_version='2.6.0',
        py_version='py312',
        env=hub,
        role=role, 
    )

    # deploy model to SageMaker Inference
    predictor = huggingface_model.deploy(
        initial_instance_count=1, # number of instances
        instance_type='ml.m5.xlarge' # ec2 instance type
    )

    predictor.predict({
        "inputs": "I like you. I love you",
    })

def main():
    print("Running deployment...")
    run_test()
    print("Exited function.")

    print("change")
    print("change")

