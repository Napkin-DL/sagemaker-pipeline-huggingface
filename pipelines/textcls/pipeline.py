"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.huggingface import HuggingFaceProcessor
from sagemaker.pytorch import PyTorchProcessor

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)

from sagemaker.huggingface import HuggingFace

from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.workflow.steps import CacheConfig
# Below we are defining a rety policy that we will apply for a number of the steps below. 
# This will make our pipeline more resiliant. If you want to know more about this, read the blog https://towardsdatascience.com/i-tried-scaling-sagemaker-pipeline-executions-and-this-happened-31279b92821e

from sagemaker.workflow.retry import (
    StepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
    SageMakerJobExceptionTypeEnum
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # s3_input_prefix = 'a2i-output'
    s3_output_prefix = 'hf_processing_output'
    
    # Here we define which exceptions to capture and when to retry the step
    step_retry_policy = StepRetryPolicy(
        exception_types=[
            StepExceptionTypeEnum.SERVICE_FAULT,
            StepExceptionTypeEnum.THROTTLING,
        ],
        backoff_rate=2.0, # the multiplier by which the retry interval increases during each attempt
        interval_seconds=30, # the number of seconds before the first retry attempt
        expire_after_mins=4*60  # keep trying for for 4 hours max
    )

    job_retry_policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        failure_reason_types=[
            SageMakerJobExceptionTypeEnum.INTERNAL_ERROR,
            SageMakerJobExceptionTypeEnum.CAPACITY_ERROR,
        ],
        backoff_rate=2.0, # the multiplier by which the retry interval increases during each attempt
        interval_seconds=30, # the number of seconds before the first retry attempt
        expire_after_mins=4*60  # keep trying for for 4 hours max
    )

    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = "ml.m5.xlarge"
    training_instance_type = "ml.g5.xlarge"
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )


    # processing step for feature engineering
    pre_processor = PyTorchProcessor(
        role=role, 
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        framework_version='1.8',
        base_job_name='PreprocessingforHF',
        sagemaker_session=pipeline_session,
    )
    processor_args = pre_processor.run(
                            code='processing-script.py',
                            source_dir='scripts',
                            outputs=[
                                ProcessingOutput(
                                    output_name='train', 
                                    source='/opt/ml/processing/output/train/',
                                    destination=f's3://{default_bucket}/{s3_output_prefix}/train'),
                                ProcessingOutput(
                                    output_name='test', 
                                    source='/opt/ml/processing/output/test/', 
                                    destination=f's3://{default_bucket}/{s3_output_prefix}/test'),
                            ]
    )

    step_process = ProcessingStep(
        name="PrepareAugmentedData", 
        step_args=processor_args,
        cache_config=cache_config
    )


    # hyperparameters, which are passed into the training job
    hyperparameters={'epochs': 1,
                     'train_batch_size': 32,
                     'model_name':'distilbert-base-uncased'
                     }

    # image_uri="763104351884.dkr.ecr.eu-west-1.amazonaws.com/huggingface-pytorch-training:1.7-transformers4.6-gpu-py36-cu110-ubuntu18.04"
    train_image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04"

    # training step for generating model artifacts
    model_path = f"s3://{default_bucket}/{s3_output_prefix}/train_result"
    
    
#     metric_definitions = [
#         {'Name': 'TrainLoss', 'Regex': r'\'loss\':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),'},
#         {'Name': 'EvalLoss', 'Regex': r'\'eval_loss\':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),'},
#         {'Name': 'EvalAcc', 'Regex': r'\'eval_accuracy\':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),'},
#         {'Name': 'EvalF1', 'Regex': r'\'eval_f1\':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),'},
#         {'Name': 'EvalPrecision', 'Regex': r'\'eval_precision\':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),'},
#         {'Name': 'EvalRecall', 'Regex': r'\'eval_recall\':([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?),'},
        
#     ]

##
    
    metric_definitions = [
        {'Name': 'TrainLoss', 'Regex': '{\'loss\': (.*?),'},
        {'Name': 'EvalLoss', 'Regex': '\'eval_loss\': (.*?),'},
        {'Name': 'EvalAcc', 'Regex': '\'eval_accuracy\': (.*?),'},
        {'Name': 'EvalF1', 'Regex': '\'eval_f1\': (.*?),'},
        {'Name': 'EvalPrecision', 'Regex': '\'eval_precision\': (.*?),'},
        {'Name': 'EvalRecall', 'Regex': '\'eval_recall\': (.*?),'},
        
    ]
    
    huggingface_estimator = HuggingFace(entry_point='train.py',
                                        source_dir='./scripts',
                                        instance_type=training_instance_type,
                                        instance_count=training_instance_count,
                                        role=role,
                                        transformers_version='4.6',
                                        pytorch_version='1.8',
                                        py_version='py36',
                                        hyperparameters = hyperparameters,
                                        metric_definitions = metric_definitions,
                                        image_uri=train_image_uri,
                                        output_path=model_path,
                                       )

    step_train = TrainingStep(
        name="HuggingFaceModelFineTune",
        estimator=huggingface_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        retry_policies=[
            step_retry_policy,
            job_retry_policy
        ],
        cache_config=cache_config
    )
    
    inf_image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04"

    step_register = RegisterModel(
        name="RegisterModel",
        estimator=huggingface_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"], # instance types recommended by data scientist to be used for real-time endpoints
        transform_instances=["ml.m5.xlarge", "ml.m5.2xlarge"], # instance types recommended by data scientist to be used for batch transform jobs
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        image_uri=inf_image_uri
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            training_instance_count,
            model_approval_status,
        ],
        steps=[step_process, step_train, step_register],
        sagemaker_session=pipeline_session,
    )
    return pipeline
