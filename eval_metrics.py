import argparse
from metrics.all_score import calculate_scores_given_paths


def eval_metrics(fake_dir, real_dir, device):
    result = calculate_scores_given_paths(
                            [fake_dir, real_dir], device=device, batch_size=50, dims=2048,
                            use_fid_inception=True, torch_svd=False)
    result = result[0]
    _, kid, fid = result
    kid_m, kid_std = kid
    da = {}
    da['FID'] = fid
    da['KID'] = kid_m
    da['KID_std'] = kid_std
    return da

parser = argparse.ArgumentParser(description='eval metrics')
parser.add_argument('-f', '--fake', type=str, help='fake images directory', required=True)
parser.add_argument('-r', '--real', type=str, help='real images directory', required=True)
parser.add_argument('-d', '--device', type=str, help='gpu or cpu device', default="cpu")

if __name__ == "__main__":
    args = parser.parse_args()
    ans = eval_metrics(args.fake, args.real, args.device)
    print(f"FID: {ans['FID']:3f}, KID: {ans['KID']:5f}, KID_std: {ans['KID_std']:8f}")