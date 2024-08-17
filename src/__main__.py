from .chapter7 import (
    custom_callbacks,
    custom_metrics,
    custom_training_loop,
    functional,
    sequential,
)
from .chapter8 import convnets
from .config import setup as setup_config


def main():
    setup_config()
    # tftest.run()
    # kerastest.setup()
    # mymodel.run()

    # sequential.main()
    # functional.main()
    # custom_metrics.main()
    custom_callbacks.main()
    # custom_training_loop.main()

    # convnets.main()


if __name__ == "__main__":
    main()
