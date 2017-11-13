/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

bool debug = false;

void print_particle(std::vector<Particle> &particles, int i) {
  std::cout << "id: " << particles[i].id << " x: " << particles[i].x
            << " y: " << particles[i].y << " theta: " << particles[i].theta
            << " weight: " << particles[i].weight << endl;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  num_particles = 50;

  std::default_random_engine gen;

  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  if (debug)
    std::cout << "INIT" << endl; // DEBUG
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = N_x(gen);
    p.y = N_y(gen);
    p.theta = N_theta(gen);
    p.weight = 1;
    particles.push_back(p);
    weights.push_back(1);
    if (debug && i < 10) {
      print_particle(particles, i);
    } // DEBUG
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add velocity and yaw rate measurements to each particle and add
  // random Gaussian noise to predict the car's position (pose). NOTE: When
  // adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;

  if (debug)
    std::cout << "PREDICT" << endl; // DEBUG
  for (int i = 0; i < num_particles; ++i) {

    double x;
    double y;
    double theta;

    if (yaw_rate) {
      x = particles[i].x + velocity / yaw_rate *
                               (sin(particles[i].theta + yaw_rate * delta_t) -
                                sin(particles[i].theta));
      y = particles[i].y + velocity / yaw_rate *
                               (cos(particles[i].theta) -
                                cos(particles[i].theta + yaw_rate * delta_t));
      theta = particles[i].theta + yaw_rate * delta_t;
    } else {
      x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      theta = particles[i].theta;
    }

    std::normal_distribution<double> N_x(x, std_pos[0]);
    std::normal_distribution<double> N_y(y, std_pos[1]);
    std::normal_distribution<double> N_theta(theta, std_pos[2]);

    particles[i].x = N_x(gen);
    particles[i].y = N_y(gen);
    particles[i].theta = N_theta(gen);
    if (debug)
      print_particle(particles, i); // DEBUG
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement (position of landmark from the map)
  // that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.

  if (debug)
    std::cout << "LANDMARK ASSOCATION" << endl; // DEBUG

  for (int i = 0; i < observations.size(); ++i) {
    double distance_min;
    if (debug)
      std::cout << "Observation: " << i << endl; // DEBUG
    distance_min = 1000.0;
    for (int j = 0; j < predicted.size(); ++j) {
      double pred_to_obs;
      pred_to_obs = dist(predicted[j].x, predicted[j].y, observations[i].x,
                         observations[i].y);
      // std::cout << "landmark: " << predicted[j].id
      //<< " distance: " << pred_to_obs << endl; // DEBUG
      if (pred_to_obs < distance_min) {
        distance_min = pred_to_obs;
        observations[i].id = predicted[j].id;
      }
    }
    if (debug)
      std::cout << "associate landmark: " << observations[i].id
                << " distance: " << distance_min << endl; // DEBUG
  }
}

int lm_index_from_id(std::vector<LandmarkObs> predicted_lm, int id) {
  for (int i = 0; i < predicted_lm.size(); ++i) {
    if (predicted_lm[i].id == id) {
      return i;
    }
  }
  return -1;
}

double multi_gauss(double x, double y, double lm_x, double lm_y, double std_x,
                   double std_y) {
  double a = pow(x - lm_x, 2.0) / (2.0 * pow(std_x, 2.0));
  double b = pow(y - lm_y, 2.0) / (2.0 * pow(std_y, 2.0));
  double p = exp(-(a + b)) / (2.0 * M_PI * std_x * std_y);
  if (debug)
    std::cout << " multi gauss:  " << p << endl; // DEBUG
  return p;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems. Keep in mind that this transformation requires
  //   both rotation AND translation (but no scaling). The following is a good
  //   resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html

  if (debug) {
    std::cout << "UPDATE WEIGHTS" << endl; // DEBUG
    std::cout << "Landmarks: " << map_landmarks.landmark_list.size() << endl;
  } // DEBUG

  for (int i = 0; i < num_particles; ++i) {
    std::vector<LandmarkObs> predicted_lm;
    std::vector<LandmarkObs> transformed_obs;

    if (debug && i < 10) {
      std::cout << "Particle " << i << endl;
    } // DEBUG

    // Create a list of landmarks within range of particle sensors
    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      double d;
      d = dist(particles[i].x, particles[i].y,
               map_landmarks.landmark_list[j].x_f,
               map_landmarks.landmark_list[j].y_f);
      if (d <= sensor_range) {
        LandmarkObs lm;
        lm.id = map_landmarks.landmark_list[j].id_i;
        lm.x = map_landmarks.landmark_list[j].x_f;
        lm.y = map_landmarks.landmark_list[j].y_f;
        predicted_lm.push_back(lm);
        if (debug) {
          std::cout << "Landmark " << j << " x: " << lm.x << " y: " << lm.y
                    << " distance " << d; // DEBUG
          std::cout << " within sensor range" << endl;
        } // DEBUG
      }
    }

    if (debug)
      std::cout << "Transform observations" << endl; // DEBUG

    // Transform observations from car coordinates to map coordinates
    for (int j = 0; j < observations.size(); ++j) {
      LandmarkObs lm;
      if (debug) {
        std::cout << "Observation " << j; // DEBUG
        std::cout << " x: " << observations[j].x << " y: " << observations[j].y
                  << endl;
      } // DEBUG
      lm.x = particles[i].x + (cos(particles[i].theta) * observations[j].x) -
             (sin(particles[i].theta) * observations[j].y);
      lm.y = particles[i].y + (sin(particles[i].theta) * observations[j].x) +
             (cos(particles[i].theta) * observations[j].y);
      transformed_obs.push_back(lm);
      if (debug) {
        std::cout << "Transformed " << j;
        std::cout << " x: " << lm.x << " y: " << lm.y << endl;
      } // DEBUG
    }

    // Associate each landmark observation with nearest map landmark
    dataAssociation(predicted_lm, transformed_obs);

    if (debug)
      std::cout << "Update weights using multi gauss" << endl; // DEBUG

    // Update particle weights with Multivariate-Gaussian Probability Density
    // Combine the probabilities to arrive at final weights (Posterior
    // Probability)
    double final_weight;
    final_weight = 1.0;
    for (int j = 0; j < transformed_obs.size(); ++j) {
      double lm_x, lm_y;
      int lm_index = lm_index_from_id(predicted_lm, transformed_obs[j].id);
      lm_x = predicted_lm[lm_index].x;
      lm_y = predicted_lm[lm_index].y;
      if (debug) {
        std::cout << "Landmark x: " << lm_x;
        std::cout << " Landmark y: " << lm_y;
      } // DEBUG
      final_weight *= multi_gauss(transformed_obs[j].x, transformed_obs[j].y,
                                  lm_x, lm_y, std_landmark[0], std_landmark[1]);
    }
    particles[i].weight = final_weight;
    weights[i] = final_weight;
    if (debug)
      std::cout << "Final weight: " << final_weight << endl; // DEBUG
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight. NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Set of resampled particles

  std::vector<Particle> resampled_p;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> d(weights.begin(), weights.end());

  for (int n = 0; n < num_particles; ++n) {
    resampled_p.push_back(particles[d(gen)]);
  }

  particles = resampled_p;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
