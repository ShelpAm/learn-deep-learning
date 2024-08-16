from .chapter7 import custom_callbacks, custom_metrics, functional, sequential
from .chapter8 import convnets
from .config import setup as setup_config


def main():
    setup_config()
    # tftest.run()
    # kerastest.setup()
    # mymodel.run()

    sequential.main()
    functional.main()
    custom_metrics.main()
    custom_callbacks.main()

    convnets.main()


if __name__ == "__main__":
    main()
