import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimAdvancedFleeBehavior extends PsyanimComponent {

    fleeBehavior;
    obstacleAvoidanceBehavior;

    sampleExpirationTime; // ms
    nSamplesForSmoothing;

    avoidanceSmoothingWindow; // ms

    constructor(entity) {

        super(entity);

        /**
         *  PsyanimAdvancedFleeBehavior is in a prototype state and really should be used
         *  with a pathfinder to get cleaner / more believable obstacle avoidance behavior.
         */

        this.sampleExpirationTime = 300;
        this.nSamplesForSmoothing = 10;

        this.avoidanceSmoothingWindow = 500;

        this._lastAvoidanceTime = 0;

        this._steeringSamples = [];
    }

    afterCreate() {

        super.afterCreate();
    }

    _addSteeringSample(steering)
    {
        if (this._steeringSamples.length >= this.nSamplesForSmoothing)
        {
            this._steeringSamples.shift();
        }

        this._steeringSamples.push({
            t: this.scene.time.now,
            steering: steering
        });
    }

    _getAverageSteeringValue() {

        let steering = new Phaser.Math.Vector2(0, 0);

        if (this._steeringSamples.length == 0)
        {
            return steering;
        }

        // compute the average value
        this._steeringSamples.forEach(sample => steering.add(sample.steering));

        if (steering.length() > this.obstacleAvoidanceBehavior.maxSeekAcceleration)
        {
            steering.setLength(this.obstacleAvoidanceBehavior.maxSeekAcceleration);
        }

        return steering;
    }

    _removeExpiredSamples() {

        if (this._steeringSamples.length == 0)
        {
            return;
        }
        
        let mostRecentSample = this._steeringSamples.at(-1);

        this._steeringSamples = this._steeringSamples
            .filter(s => mostRecentSample.t - s.t < this.sampleExpirationTime);
    }

    getSteering(target) {

        this._removeExpiredSamples();

        let steering = this.obstacleAvoidanceBehavior.getSteering();

        if (steering.length() > 1e-3)
        {
            this._lastAvoidanceTime = this.scene.time.now;
        }
        else
        {
            steering = this.fleeBehavior.getSteering(target);
        }

        let timeSinceLastAvoidance = this.scene.time.now - this._lastAvoidanceTime;

        if (timeSinceLastAvoidance < this.avoidanceSmoothingWindow)
        {
            this._addSteeringSample(steering);

            return this._getAverageSteeringValue();
        }
        else
        {
            return steering;
        }
    }
}