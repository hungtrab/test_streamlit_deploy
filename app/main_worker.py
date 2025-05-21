# app/main_worker.py
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import pytz
import os # For AM_I_IN_A_DOCKER_CONTAINER check if needed for other things

# Use relative imports, assuming this is run as part of the 'app' package
try:
    from .config import (
        SCHEDULER_TIMEZONE, DATA_UPDATE_HOUR_ET, DATA_UPDATE_MINUTE_ET,
        PREDICTION_HOUR_ET, PREDICTION_MINUTE_ET
    )
    from .scheduler_tasks import (
        daily_data_ingestion_and_db_update_job,
        daily_prediction_trigger_job
        # weekly_model_retraining_job # Uncomment if you implement and want to schedule it
    )
except ImportError:
    print("MAIN_WORKER: Error during relative import. Ensure PYTHONPATH is correct or run as part of a package.")
    # Fallback for potential direct execution (less ideal)
    from config import (
        SCHEDULER_TIMEZONE, DATA_UPDATE_HOUR_ET, DATA_UPDATE_MINUTE_ET,
        PREDICTION_HOUR_ET, PREDICTION_MINUTE_ET
    )
    from scheduler_tasks import (
        daily_data_ingestion_and_db_update_job,
        daily_prediction_trigger_job
        # weekly_model_retraining_job
    )

if __name__ == "__main__":
    # Ensure that AM_I_IN_A_DOCKER_CONTAINER is set if config.py's path helpers rely on it
    # This is usually set in docker-compose.yml for the worker service.
    # For local testing of this script, you might need to set it manually:
    # os.environ["AM_I_IN_A_DOCKER_CONTAINER"] = "false" # Or "true" if testing Docker-like paths locally

    scheduler = BlockingScheduler(timezone=pytz.timezone(SCHEDULER_TIMEZONE))
    print(f"MAIN_WORKER: Scheduler initialized with timezone {SCHEDULER_TIMEZONE}.")

    # 1. Job for Daily Data Ingestion and DB Price Updates
    # This job will fetch data for ALL tickers and update their prices in the DB.
    scheduler.add_job(
        daily_data_ingestion_and_db_update_job,
        trigger=CronTrigger(
            hour=DATA_UPDATE_HOUR_ET,
            minute=DATA_UPDATE_MINUTE_ET,
            timezone=SCHEDULER_TIMEZONE # APScheduler uses this timezone for the trigger
        ),
        id='daily_data_ingestion_and_db_update_job',
        name='Daily Full Data Ingestion and Price DB Update',
        replace_existing=True,
        misfire_grace_time=3600 # Allow job to run up to 1 hour late if scheduler was down
    )
    print(f"MAIN_WORKER: Scheduled 'Daily Full Data Ingestion and Price DB Update' job "
          f"at {DATA_UPDATE_HOUR_ET:02d}:{DATA_UPDATE_MINUTE_ET:02d} {SCHEDULER_TIMEZONE}.")

    # 2. Job for Triggering Daily Predictions (after data update)
    # This job will trigger predictions for ALL tickers and ALL relevant models via API.
    scheduler.add_job(
        daily_prediction_trigger_job,
        trigger=CronTrigger(
            hour=PREDICTION_HOUR_ET,
            minute=PREDICTION_MINUTE_ET,
            timezone=SCHEDULER_TIMEZONE
        ),
        id='daily_prediction_trigger_job_all_models',
        name='Daily Prediction Trigger for All Tickers & Models',
        replace_existing=True,
        misfire_grace_time=3600
    )
    print(f"MAIN_WORKER: Scheduled 'Daily Prediction Trigger for All Tickers & Models' job "
          f"at {PREDICTION_HOUR_ET:02d}:{PREDICTION_MINUTE_ET:02d} {SCHEDULER_TIMEZONE}.")

    # 3. Optional: Job for Weekly Model Retraining (Placeholder)
    # scheduler.add_job(
    #     weekly_model_retraining_job,
    #     trigger=CronTrigger(day_of_week='sun', hour=2, minute=0, timezone=SCHEDULER_TIMEZONE),
    #     id='weekly_model_retraining_job',
    #     name='Weekly Model Retraining for All Tickers',
    #     replace_existing=True,
    #     misfire_grace_time=3600 * 6 # Allow to run up to 6 hours late
    # )
    # print(f"MAIN_WORKER: Scheduled 'Weekly Model Retraining' job for Sunday 02:00 {SCHEDULER_TIMEZONE}.")


    print(f"MAIN_WORKER: [{datetime.now()}] Scheduler starting. Press Ctrl+C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("MAIN_WORKER: Scheduler stopped by user.")
    finally:
        if scheduler.running:
            scheduler.shutdown()
        print("MAIN_WORKER: Scheduler shut down.")