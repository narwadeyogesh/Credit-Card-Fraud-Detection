from CreditCardFraud.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CreditCardFraud.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from CreditCardFraud.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from CreditCardFraud.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from CreditCardFraud.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from CreditCardFraud.logging import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<<")
    model_evaluator = ModelEvaluationPipeline()
    model_evaluator.main()
    logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e