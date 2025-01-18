from xcpetion import build_xception_backbone
from models import get_xception_based_model
from utils import get_nof_params
from xcpetion import disable_ssl_verification

def main():
    model1 = build_xception_backbone()
    print(f"The number of parameters in the Xception model is {get_nof_params(model1)}")

    # uncomment only if the weights are not already in pytorch cache and the ssl certificate causes problems
    # disable_ssl_verification()

    model2 = get_xception_based_model()
    print(f"The number of parameters in the Xception model with the new head is {get_nof_params(model2)}")

if __name__ == '__main__':
    main()