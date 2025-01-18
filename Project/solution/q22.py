from xcpetion import build_xception_backbone
from utils import get_nof_params

def main():
    model = build_xception_backbone()
    print(f"The number of parameters in the Xception model is {get_nof_params(model)}")

if __name__ == '__main__':
    main()