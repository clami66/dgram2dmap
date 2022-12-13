import glob
import pickle
import string
import logging
import argparse
from sys import argv, exit
from copy import copy
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from Bio.PDB import *


def add_arguments(parser):
    parser.add_argument(
        "in_folder",
        help="AlphaFold model output folder",
    )
    parser.add_argument(
        "--maxD",
        help="Maximum distance (in Å) for constraints output",
        default=20.0,
        type=float,
        required=False,
        metavar="20.0",
    )
    parser.add_argument(
        "--limits",
        help="Select a 'patch' of constraints between two subsets of residues (e.g. 0:100 200:300)",
        nargs=2,
        required=False,
        metavar=("i:j", "k:l"),
    )
    parser.add_argument(
        "--chains",
        help="Extract constraints between two chains (e.g. A B)",
        nargs=2,
        required=False,
        metavar=("chain1", "chain2"),
    )
    parser.add_argument(
        "--plot",
        help="Plot the distances with bounding boxes",
        action="store_true",
    )
    parser.add_argument(
        "--argmax",
        help="Use argmax to find the most likely distance instead of interpolating",
        action="store_true",
    )
    parser.add_argument(
        "--rosetta",
        help="Export below-threshold (see maxD) distances in a Rosetta constraint files",
        action="store_true",
    )
    parser.add_argument(
        "--pdb",
        help="PDB model of the target protein",
        required=False,
        metavar="ranked_0.pdb",
    )
    parser.add_argument(
        "--rosetta_format",
        help="Format of rosetta constraint function with placeholder {} for distances",
        required=False,
        metavar="'HARMONIC {} 1.0'",
        default="'HARMONIC {:.2f} 1.0 TAG'",
    )


def get_distance_predictions(results, interpolate=True):
    bin_edges = results["distogram"]["bin_edges"]
    bin_edges = np.insert(bin_edges, 0, 0)

    if interpolate:
        distogram_softmax = softmax(results["distogram"]["logits"], axis=2)
        distance_predictions = np.sum(np.multiply(distogram_softmax, bin_edges), axis=2)
    else:  # pick maximum probability distance instead
        distogram_argmax = np.argmax(
            results["distogram"]["logits"][:, :, :63], axis=2
        )  # skips last bin to avoid being too conservative
        distance_predictions = bin_edges[distogram_argmax]

    return distance_predictions


def get_chain_limits(features):
    chain_limits = {"A": (1, features["msa"].shape[1])}

    if "asym_id" in features:  # then it's a multimer
        chain_ids = features["asym_id"].astype("int")
        for i in range(chain_ids[-1]):
            chain_starts = np.where(chain_ids == i + 1)[0][0] + 1

            chain_stops = np.where(chain_ids == i + 1)[0][-1] + 1

            chain_limits[string.ascii_uppercase[i]] = (chain_starts, chain_stops)
    return chain_limits


def get_bounding_boxes(limitA, limitB, color="r"):
    rect1 = patches.Rectangle(
        (limitA[0] - 1, limitB[0] - 1),
        limitA[1] - limitA[0],
        limitB[1] - limitB[0],
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    rect2 = patches.Rectangle(
        (limitB[0] - 1, limitA[0] - 1),
        limitB[1] - limitB[0],
        limitA[1] - limitA[0],
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    return rect1, rect2


def load_features(filepath):
    with open(filepath, "rb") as p:
        features = pickle.load(p)

    return features


def load_results(filepath, interpolate=True):

    with open(filepath, "rb") as p:
        results = pickle.load(p)
        distance_predictions = get_distance_predictions(results, interpolate)
        if "predicted_aligned_error" in results:
            pae = results["predicted_aligned_error"]
        else:
            pae = None
    return distance_predictions, pae


def get_rosetta_constraints(
    dist_matrix,
    func_format="HARMONIC {:.2f} 1.0 TAG",
    atom_name="CA",
    maxD=20.0,
    SD=1.0,
    chain1=None,
    chain2=None,
    limitA=None,
    limitB=None,
):
    constraints = []
    if limitA is not None and limitB is not None:
        x0 = limitA[0]
        x1 = limitA[1]
        y0 = limitB[0]
        y1 = limitB[1]
    else:
        x0 = 0
        y0 = 10
        x1 = y1 = dist_matrix.shape[0]

    for i in range(x0, x1):
        for j in range(y0, y1):
            dist = 0.5 * (dist_matrix[i, j] + dist_matrix[j, i])
            if dist < maxD:
                function = func_format.format(dist)
                if chain1 and chain2:
                    constraints.append(
                        f"AtomPair {atom_name} {i+1-x0}{chain1} {atom_name} {j+1-y0}{chain2} {function}\n"
                    )
                else:
                    constraints.append(
                        f"AtomPair {atom_name} {i+1} {atom_name} {j+1} {function}\n"
                    )
    return constraints


def compare_to_native(
    filepath,
    pdb_path,
    predicted_distances,
    limitA=None,
    limitB=None,
):

    parser = PDBParser()
    structure = parser.get_structure("model", pdb_path)

    compared_dist = np.triu(predicted_distances)

    c_alphas = [r["CB"] if "CB" in r else r["CA"] for r in structure.get_residues()]

    for i, ca_i in enumerate(c_alphas):
        for j, ca_j in enumerate(c_alphas):
            if j < i:
                dist = ca_i - ca_j
                compared_dist[i, j] = min(dist, 22)

    if limitA and limitB:
        predicted_distances = predicted_distances[
            limitA[0] - 1 : limitA[1], limitB[0] - 1 : limitB[1]
        ]
        real_dist = np.zeros_like(predicted_distances)
        real_dist = np.transpose(
            compared_dist[limitB[0] - 1 : limitB[1], limitA[0] - 1 : limitA[1]]
        )
    else:
        predicted_distances = np.triu(predicted_distances)
        real_dist = np.transpose(np.tril(compared_dist))

    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.9]})
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )

    d = ax[0].imshow(compared_dist)
    plt.colorbar(d, ax=ax[0], fraction=0.046, pad=0.04)
    ax[0].set_ylabel("← PDB Model")
    ax[0].set_xlabel("Distogram →")
    ax[0].xaxis.set_label_position("top")
    ax[0].set_yticks([])
    ax[1].set_xlim(0, 22)
    ax[1].set_ylim(0, 22)
    ax[1].set_box_aspect(1)
    ax[1].scatter(real_dist.reshape(-1), predicted_distances.reshape(-1))
    lims = [
        np.min([ax[1].get_xlim(), ax[1].get_ylim()]),  # min of both axes
        np.max([ax[1].get_xlim(), ax[1].get_ylim()]),  # max of both axes
    ]
    ax[1].plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax[1].set_xlabel("PDB model distances")
    ax[1].set_ylabel("Distogram distances")

    if limitA and limitB:
        # plots a bounding box if any
        rect1, rect2 = get_bounding_boxes(limitA, limitB)
        ax[0].add_patch(rect1)
        ax[0].add_patch(rect2)

    ax[0].set_title("Distance map comparison")
    ax[1].set_title("Distance agreement\n(bounded area)")

    plt.savefig(filepath, dpi=600)
    plt.close()


def plot_distances(filepath, distances, pae=None, limitA=None, limitB=None):

    if pae is not None:
        fig, ax = plt.subplots(1, 2)
        d = ax[0].imshow(distances)
        plt.colorbar(d, ax=ax[0], fraction=0.046, pad=0.04)
        p = ax[1].imshow(pae)
        plt.colorbar(p, ax=ax[1], fraction=0.046, pad=0.04)
        ax[0].title.set_text("Distance map")
        ax[1].title.set_text("Predicted aligned error")
        ax[1].set_yticks([])
    else:
        fig, ax = plt.subplots()
        d = ax.imshow(distances)
        plt.colorbar(d, ax=ax, fraction=0.046, pad=0.04)
        ax.title.set_text("Distance map")

    if limitA and limitB:
        # plots a bounding box if any
        rect1, rect2 = get_bounding_boxes(limitA, limitB)

        if pae is not None:
            rect3 = copy(rect1)
            rect4 = copy(rect2)
            ax[0].add_patch(rect1)
            ax[0].add_patch(rect2)
            ax[1].add_patch(rect3)
            ax[1].add_patch(rect4)
        else:
            ax.add_patch(rect1)
            ax.add_patch(rect2)
    
    plt.savefig(filepath, dpi=600)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and format distance constraints from AlphaFold distograms"
    )
    add_arguments(parser)
    args = parser.parse_args()

    features = load_features(f"{args.in_folder}/features.pkl")

    limitA = limitB = None
    chain1 = chain2 = None

    if args.limits:
        limitA = [int(l) for l in args.limits[0].split(":")]
        limitB = [int(l) for l in args.limits[1].split(":")]
    elif args.chains:
        chain_limits = get_chain_limits(features)
        chain1, chain2 = args.chains
        limitA = chain_limits[chain1]
        limitB = chain_limits[chain2]

    pickle_list = glob.glob(f"{args.in_folder}/result_*.pkl")
    interpolate = False if args.argmax else True

    for i, pickle_output in enumerate(pickle_list):
        logging.warning(
            f"Processing pickle file {i+1}/{len(pickle_list)}: {pickle_output}"
        )
        dist, pae = load_results(pickle_output, interpolate=interpolate)
        np.savetxt(f"{pickle_output}.dmap", dist)

        if args.plot:
            plot_distances(f"{pickle_output}.dmap.png", dist, pae, limitA, limitB)

        if args.rosetta:
            rosetta_constraints = get_rosetta_constraints(
                dist,
                func_format=args.rosetta_format,
                maxD=args.maxD,
                chain1=chain1,
                chain2=chain2,
                limitA=limitA,
                limitB=limitB,
            )
            with open(f"{pickle_output}.rosetta_constraints", "w") as out:
                for line in rosetta_constraints:
                    out.write(line)

        if args.pdb:
            compare_to_native(
                f"{pickle_output}.agreement.png",
                args.pdb,
                dist,
                limitA,
                limitB,
            )


if __name__ == "__main__":
    exit(main())
