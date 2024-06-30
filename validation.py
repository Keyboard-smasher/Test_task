import matplotlib.pyplot as plt
import torch
from PIL import Image
from training_classificator import SiameseModel, SiameseDataset, preprocessing_for_resnet50
from matplotlib import pyplot
from pathlib import Path


def load_model(name='./best_V1.pt'):
    """
    Загрузить обученную модель
    """
    state_dict = torch.load(name)
    model = SiameseModel()
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    return model


def model_forward(model, image1_path, image2_path, probability_thr=0.03, number=0, workdir=None):
    """
    DO NOT try PNG photos
    Сделать предсказание по 2м изображениям
    """
    preprocessing = preprocessing_for_resnet50()
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    fg, ax = pyplot.subplots(1, 2)
    ax[0].imshow(image1)
    ax[0].set_title('Image 1')
    ax[1].imshow(image2)
    ax[1].set_title('Image 2')

    image1 = preprocessing(image1)
    image2 = preprocessing(image2)

    probability = model.forward(image1.unsqueeze(0), image2.unsqueeze(0))
    probability = probability.detach().numpy().squeeze()
    out = f'This is the same person ' if probability > probability_thr else \
        f'This is not the same person'
    print(out)
    fg.suptitle(out)
    if workdir is not None:
        pyplot.savefig(workdir / f'{number + 2}.png')
    pyplot.show()


def process_dir(model, dr: Path):
    """
    Функция для обработки папок test1 и test2
    """
    reference = dr / 'ref.jpg'
    targets = dr / 'targets'
    for i, image in enumerate(targets.glob('*.jpg')):
        model_forward(model, reference, image, number=i, workdir = dr / 'out')


if __name__ == '__main__':
    """
    Запуск обработки папок или отдельных изображений
    """
    model = load_model()
    # image1 = Path(__file__).parent / 'example' / 'Adisai_Bodharamik_0001.jpg'
    # image2 = Path(__file__).parent / 'example' / 'Adriana_Lima_0001.jpg'
    # image1 = Path(__file__).parent / 'example' / 'Zoran_Djindjic_0001.jpg'
    # image2 = Path(__file__).parent / 'example' / 'Zoran_Djindjic_0002.jpg'
    # image1 = Path(__file__).parent / 'example' / 'Zurab_Tsereteli_0001.jpg'
    # image2 = Path(__file__).parent / 'example' / 'Zydrunas_Ilgauskas_0001.jpg'
    # image1 = Path(__file__).parent / 'test1' / 'targets' / 'vedmak2_00.jpg'
    # image2 = Path(__file__).parent / 'test1' / 'targets' / 'vedmak3_00.jpg'
    # model_forward(model, image1, image2)
    process_dir(model, Path(__file__).parent / 'test1')