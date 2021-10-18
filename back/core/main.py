from core import process


def c_main(path, model, ext):
    image_data = process.pre_process(path)
    image_info = model.detect(path)

    return image_data[1] + '.' + ext, image_info


if __name__ == '__main__':
    pass
