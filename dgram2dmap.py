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
        "in_folder", help="AlphaFold model output folder",
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
        "--plot", help="Plot the distances with bounding boxes", action="store_true",
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


def get_distance_predictions(results):
    bin_edges = results["distogram"]["bin_edges"]
    bin_edges = np.insert(bin_edges, 0, 0)

    distogram_softmax = softmax(results["distogram"]["logits"], axis=2)
    distance_predictions = np.sum(np.multiply(distogram_softmax, bin_edges), axis=2)

    return distance_predictions


def load_features(filepath):
    with open(filepath, "rb") as p:
        features = pickle.load(p)

    return features


def load_results(filepath):

    with open(filepath, "rb") as p:
        results = pickle.load(p)
        distance_predictions = get_distance_predictions(results)
        if "predicted_aligned_error" in results:
            pae = results["predicted_aligned_error"]
        else:
            pae = None
    return distance_predictions, pae


def get_rosetta_constraints(
    dist_matrix,
    func_type="HARMONIC",
    atom_name="CA",
    maxD=20.0,
    SD=1.0,
    chain1="A",
    chain2="A",
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
                constraints.append(
                    f"AtomPair {atom_name} {i+1}{chain1} {atom_name} {j+1}{chain2} {func_type} {dist:.2f} {SD} TAG\n"
                )
    return constraints


def compare_to_native(filepath, pdb_path, predicted_distances):

    parser = PDBParser()
    structure = parser.get_structure("model", pdb_path)

    real_dist = np.zeros_like(predicted_distances)

    for i in range(1, predicted_distances.shape[0] + 1):
        atom1 = structure[0]["A"][i]["CA"]
        for j in range(i, predicted_distances.shape[0] + 1):
            atom2 = structure[0]["A"][j]["CA"]
            real_dist[i - 1, j - 1] = real_dist[j - 1, i - 1] = atom1 - atom2

    fig1, ax = plt.subplots()

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_box_aspect(1)
    plt.scatter(real_dist.reshape(-1), predicted_distances.reshape(-1))
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax.title.set_text("Distogram-model distance agreement")
    ax.set_xlabel("Model Ca-Ca distances")
    ax.set_ylabel("Distogram converted distances")
    plt.savefig(filepath)


def get_chain_limits(features):
    chain_limits = {"A": (1, features["msa"].shape[1])}

    if "asym_id" in features:  # then it's a multimer
        chain_ids = features["asym_id"].astype("int")
        for i in range(chain_ids[-1]):
            chain_starts = np.where(chain_ids == i + 1)[0][0] + 1

            chain_stops = np.where(chain_ids == i + 1)[0][-1] + 1

            chain_limits[string.ascii_uppercase[i]] = (chain_starts, chain_stops)
    return chain_limits


def plot_distances(filepath, distances, pae=None, limitA=None, limitB=None):

    if pae is not None:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(distances)
        ax[1].imshow(pae)
        ax[0].title.set_text("Distance map")
        ax[1].title.set_text("Predicted aligned error")
    else:
        fig, ax = plt.subplots()
        ax.title.set_text("Distance map")

    if limitA and limitB:
        # plots a bounding box if any
        rect1 = patches.Rectangle(
            (limitA[0] - 1, limitB[0] - 1),
            limitA[1] - limitA[0],
            limitB[1] - limitB[0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        rect2 = patches.Rectangle(
            (limitB[0] - 1, limitA[0] - 1),
            limitB[1] - limitB[0],
            limitA[1] - limitA[0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

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
    plt.savefig(filepath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract and format distance constraints from AlphaFold distograms"
    )
    add_arguments(parser)
    args = parser.parse_args()

    features = load_features(f"{args.in_folder}/features.pkl")

    limitA = limitB = None
    chain1 = chain2 = "A"

    if args.limits:
        limitA = [int(l) for l in args.limits[0].split(":")]
        limitB = [int(l) for l in args.limits[1].split(":")]
    elif args.chains:
        chain_limits = get_chain_limits(features)
        chain1, chain2 = args.chains
        limitA = chain_limits[chain1]
        limitB = chain_limits[chain2]

    pickle_list = glob.glob(f"{args.in_folder}/result_*.pkl")

    for i, pickle_output in enumerate(pickle_list):
        logging.warning(
            f"Processing pickle file {i+1}/{len(pickle_list)}: {pickle_output}"
        )
        dist, pae = load_results(pickle_output)
        np.savetxt(f"{pickle_output}.dmap", dist)

        if args.plot:
            plot_distances(f"{pickle_output}.dmap.png", dist, pae, limitA, limitB)

        if args.rosetta:
            rosetta_constraints = get_rosetta_constraints(
                dist,
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
            compare_to_native(f"{pickle_output}.agreement.png", args.pdb, dist)


if __name__ == "__main__":
    exit(main())
