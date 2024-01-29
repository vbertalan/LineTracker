"""Allow to visualize/convert logs: List[str], clustering: Dict[int, int] to console print or markdown/hmtl text"""

from typing import *
import colorsys
import rich.console as c


class ColoredClustering(NamedTuple):
    """
    - text: str, the text
    - color: str, the color chosen (rgb(...,...,...)) for the line
    - cluster: int, the cluster chosen for the line
    """

    text: str
    color: str
    cluster: int


def generate_hsv_palette(
    num_colors: int, saturation: float = 1.0, value: float = 1.0
) -> List[Tuple[int, int, int]]:
    """From a number of colors, it generates a list of the num_colors colors with the rgb code for each

    # Arguments
    - num_colors: int, the number of colors to generate
    - saturation: float=1.0, the constant saturation to use
    - value: float=1.0, the constant value to use

    # Returns
    - List[Tuple[int,int,int]], [(color1_r, color1_g, color1_b), (color2_r, color2_g, color2_b), ...] list of colors generated (num_colors)
    """
    colors = []
    hue_step = 1.0 / num_colors

    for i in range(num_colors):
        hue = i * hue_step
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgb = tuple(int(x * 255) for x in rgb)
        colors.append(rgb)

    return colors


def convert_clustering_to_colored_clustering(
    logs: List[str], clustering: Dict[int, int]
) -> List[ColoredClustering]:
    """Convert a clustering of texts lines to a ColoredClustering

    # Arguments
    - logs: List[str], list of texts for each line
    - clustering: Dict[int, int], mapping from the line number to the cluster chosen

    # Return
    - str, the html code of the visualization
    """
    unique_clusters = set(c for _, c in clustering.items())
    cluster_color_mapping = {
        cluster: color
        for cluster, color in zip(
            unique_clusters, generate_hsv_palette(len(unique_clusters))
        )
    }
    L = []
    for i, text in enumerate(logs):
        r, g, b = cluster_color_mapping[clustering[i]]
        L.append(
            ColoredClustering(
                text=text, color=f"rgb({r},{g},{b})", cluster=clustering[i]
            )
        )
    return L


def print_colored_clustering_rich(
    lines: List[ColoredClustering], force_jupyter: bool = False
):
    """Print in the terminal or jupyter notebood a clustering result where the colors for each line has already been computed

    # Arguments
    - lines: List[ColoredClustering], the colored lines with their clusters associated (see doc of ColoredClustering)
    - force_jupyter: bool = False, if used in a notebook to pass to console.print of rich (see https://github.com/Textualize/rich/blob/fd981823644ccf50d685ac9c0cfe8e1e56c9dd35/rich/console.py#L1624)
    
    """
    # Initialize rich console
    console = c.Console(
        color_system="auto", highlight=False, force_jupyter=force_jupyter
    )
    console.print()
    for line_id, (text, color, cluster) in enumerate(lines):
        # print format is "line_number-cluster_id: text of the log"
        console.print(f"{line_id:03d}-{cluster}: {text}", style=color, end="\n")
    console.print()

def print_colored_paired_clustering_rich(
    lines1: List[ColoredClustering], lines2: List[ColoredClustering], force_jupyter: bool = False
):
    """Print in the terminal or jupyter notebood a clustering result where the colors for each line has already been computed

    # Arguments
    - lines: List[ColoredClustering], the colored lines with their clusters associated (see doc of ColoredClustering)
    - force_jupyter: bool = False, if used in a notebook to pass to console.print of rich (see https://github.com/Textualize/rich/blob/fd981823644ccf50d685ac9c0cfe8e1e56c9dd35/rich/console.py#L1624)
    
    """
    # Initialize rich console
    console = c.Console(
        color_system="auto", highlight=False, force_jupyter=force_jupyter
    )
    console.print()
    for line_id, ((text, color1, cluster1),(_, color2, cluster2)) in enumerate(zip(lines1,lines2)):
        # print format is "line_number-cluster_id: text of the log"
        console.print(f"{line_id:03d}-{cluster1}-{cluster2}: {text}", style=f"{color1} on {color2}", end="\n")
    console.print()

def generate_clustering_markdown_html(lines: List[ColoredClustering]) -> str:
    """Convert each colored clustered line provided to a markdown/html text where each cluster has a color in the format
    
    ```
    line_number1-cluster_id1: <span style='color:...'>text of the log1</span>
    line_number2-cluster_id2: <span style='color:...'>text of the log1</span>2
    ```
    
    # Arguments
    - lines: List[ColoredClustering], the colored lines with their clusters associated (see doc of ColoredClustering)
    
    # Returns
    - str, the markdown full text as described above
    
    """
    html = []
    for line_id, (text, color, cluster) in enumerate(lines):
        html.append(f"{line_id:03d}-{cluster}: <span style='color:{color}'>{text}</span>")
    return "\n\n".join(html)
