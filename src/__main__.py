from .config import setup as setup_config
from .chapter7 import custom_metrics, functional, sequential
from .chapter8 import convnets


def main():
    setup_config()
    # tftest.run()
    # kerastest.setup()
    # mymodel.run()

    sequential.main()
    functional.main()
    custom_metrics.main()

    convnets.main()


if __name__ == "__main__":
    main()
