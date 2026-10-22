# Third-party dependencies and attribution

- Setup and dataset documentation reference
  [GIMO](https://github.com/y-zheng18/GIMO) and
  [SIF3D](https://github.com/kjle6/SIF3D-master), including their coordinate
  conventions and PointNet++ compatibility instructions. The GIMO dataset is
  obtained through its [official access process](https://github.com/y-zheng18/GIMO#dataset).
  Raw and processed dataset files are not redistributed here.

- `models/pointnet_plus2.py` acknowledges code from
  [Erik Wijmans' Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
  Install that project's Python package and compiled operations separately.
- SMPL-X and VPoser assets and software are provided by their respective
  upstream projects: [SMPL-X](https://smpl-x.is.tue.mpg.de/) and
  [human_body_prior](https://github.com/nghorbani/human_body_prior).
  Their model assets and the research VPoser snapshot are not redistributed.
- Dataset assets and prepared scene/object caches are external inputs.
  Obtain them from their providers and follow their terms.

Existing attribution in source files is preserved. No new repository-wide
license is assigned by this source-packaging step.
