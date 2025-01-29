import Phaser from 'phaser';

    import { PsyanimComponent } from 'psyanim2';

    export default class MyFirstMovementComponent extends PsyanimComponent {

        speed;

        constructor(entity) {

            super(entity);

            this.speed = 0.5;
        }

        update(t, dt) {

            super.update(t, dt);

            let oldPosition = this.entity.position;

            // compute speed
            if (oldPosition.x > 700 && this.speed > 0)
            {
                this.speed *= -1.0;
            }
            else if (oldPosition.x < 100 && this.speed < 0)
            {
                this.speed *= -1.0;
            }

            // compute displacement over 'dt'
            let displacement = new Phaser.Math.Vector2(this.speed * dt, 0);

            // compute new position and set this entity's position to it
            oldPosition.add(displacement);

            this.entity.position = oldPosition;
        }
    }