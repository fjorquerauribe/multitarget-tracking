# Multitarget tracking

A multi target tracker based on Gaussian Mixture Probability Density Hypothesis Density Filter using Determinantal Point Processes observations.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- CMake 2.8 or newer
- OpenCV 3.0
- Eigen 3.2

### Installating

Download the project:
```
- git clone https://github.com/fjorquerauribe/multitarget-tracking.git
```

Build the project:
```
- cd multitarget-tracking
- mkdir build
- cmake ..
- make
```

## Running the test

Download the MOT Challenge datasets https://motchallenge.net/

Create a symbolic link to the MOT Challenge datasets folder:
```
- ln -s path/to/datasets/ data
```

Copy script to run Gaussian Mixture PHD filter:
```
- cp ../scripts/start_gm_phd.sh .
```

Example: run Gaussian Mixture PHD filter over MOT16-02 sequence with public detections:
```
- ./start_gm_phd.sh MOT16 train MOT16-02 public 1
```

## Authors

* **Felipe Jorquera Uribe** - [fjorquerauribe](https://github.com/fjorquerauribe)
* **Sergio Hernández** - [shernandez](https://github.com/sherna90)

See also the list of [contributors](https://github.com/fjorquerauribe/multitarget-tracking/graphs/contributors) who participated in this project.

## License

This project is licensed under the Apache 2 license - see the [LICENSE.md](LICENSE.md) file for details