from database.model import MainVideo


def main():
    ts15_pks = list(MainVideo.all_pks())

    for pk in ts15_pks:
        stories = MainVideo.get(pk).stories


if __name__ == "__main__":
    main()
