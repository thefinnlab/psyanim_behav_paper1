import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimArriveBehavior extends PsyanimComponent {

    maxSpeed;
    maxAcceleration;

    innerDecelerationRadius;
    outerDecelerationRadius;

    constructor(entity) {

        super(entity);

        this.maxSpeed = 5;
        this.maxAcceleration = 0.2;
    
        this.innerDecelerationRadius = 25;
        this.outerDecelerationRadius = 140;
    }

    getSteering(target) {

        let currentPosition = new Phaser.Math.Vector2(this.entity.x, this.entity.y);

        let targetRelativePosition = new Phaser.Math.Vector2(target.x, target.y);
        targetRelativePosition.subtract(currentPosition);

        let r = targetRelativePosition.length();

        let desiredSpeed = 0;

        let scaledMaxAcceleration = this.maxAcceleration;

        if (r <= this.innerDecelerationRadius)
        {
            this.entity.setVelocity(0, 0);

            return new Phaser.Math.Vector2(0, 0);
        }
        else if (r > this.outerDecelerationRadius)
        {
            desiredSpeed = this.maxSpeed;
        }
        else // ((r > this.innerDecelerationRadius) && r < this.r2)
        {
            //  v(r) = ((v_max) / (r2 - r1)) * (r - r1)
            desiredSpeed = ((this.maxSpeed) / (this.outerDecelerationRadius - this.innerDecelerationRadius)) * (r - this.innerDecelerationRadius);
            scaledMaxAcceleration = (r - this.innerDecelerationRadius) / (this.outerDecelerationRadius - this.innerDecelerationRadius) * this.maxAcceleration;
        }

        let desiredVelocity = targetRelativePosition.clone();

        desiredVelocity.setLength(desiredSpeed);

        let currentVelocityXY = this.entity.getVelocity();

        let currentVelocity = new Phaser.Math.Vector2(currentVelocityXY.x, currentVelocityXY.y);

        let acceleration = desiredVelocity.clone();
        acceleration.subtract(currentVelocity);

        if (acceleration.length() > scaledMaxAcceleration)
        {
            acceleration.setLength(scaledMaxAcceleration);
        }

        return acceleration;
    }
}