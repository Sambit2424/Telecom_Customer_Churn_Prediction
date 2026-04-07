from telecom_churn.logger import LoggerFactory


def test_logger_factory_creates_logger():
    logger = LoggerFactory.get_logger("telecom_churn_test")
    assert logger.name == "telecom_churn_test"
    assert logger.handlers
