# import tftest
# import kerastest
import config
import chapter7.sequential
import chapter7.functionalapi


def main():
    config.setup()
    # tftest.run()
    # kerastest.setup()
    # mymodel.run()
    print("chapter7.seqmodel: main")
    chapter7.sequential.main()
    print("\nchapter7.functionalapi: main")
    chapter7.functionalapi.main()


if __name__ == "__main__":
    main()
