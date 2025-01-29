import Phaser from 'phaser';

import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimEvadeBehavior extends PsyanimComponent {

    maxPredictionTime;

    fleeBehavior;

    constructor(entity) {

        super(entity);

        this.maxPredictionTime = 1.0;
    }

    getSteering(target) {

        let targetPosition = target.position;
        let distanceToTarget = this.entity.position.subtract(targetPosition).length();

        let targetVelocity = target.velocity;
        let targetSpeed = targetVelocity.length();

        let predictionTime = this.maxPredictionTime;

        if (targetSpeed > distanceToTarget / this.maxPredictionTime)
        {
            predictionTime = distanceToTarget / targetSpeed;

            // don't project too far out, predicted position can go past character.
            // Unity Movement AI package uses this magic number.  can adjust if needed.
            predictionTime *= 0.9;
        }

        let evadeTargetPosition = targetVelocity.clone();
        evadeTargetPosition.scale(predictionTime);
        evadeTargetPosition.add(targetPosition);

        let evadeTarget = {
            position: evadeTargetPosition,
        }

        return this.fleeBehavior.getSteering(evadeTarget);
    }
}