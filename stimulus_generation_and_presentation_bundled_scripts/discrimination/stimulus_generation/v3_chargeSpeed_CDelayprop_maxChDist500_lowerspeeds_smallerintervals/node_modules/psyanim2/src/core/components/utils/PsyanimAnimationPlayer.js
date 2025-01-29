import PsyanimComponent from '../../PsyanimComponent.js';

export default class PsyanimAnimationPlayer extends PsyanimComponent {

    constructor(entity) {

        super(entity);

        this._clip = null;
        this._currentIndex = 0;

        this.events = new Phaser.Events.EventEmitter();
    }

    play(animationClip) {

        this._clip = animationClip;
        this._currentIndex = 0;
        this._nSamples = animationClip.getSampleCount();

        this.entity.setPhysicsEnabled(false);
    }

    stop() {

        this._clip = null;
        this._currentIndex = 0;
        this._nSamples = 0;
    }

    destroy() {

        this.stop();

        super.destroy();
    }

    update(t, dt) {

        super.update(t, dt);

        if (this._clip != null)
        {
            if (this._currentIndex < this._nSamples)
            {
                let currentSample = this._clip.getSample(this._currentIndex);

                this.entity.position = new Phaser.Math.Vector2(
                    currentSample.x, currentSample.y);

                this.entity.rotation = currentSample.rotation;

                this._currentIndex++;
            }
            else
            {
                this._clip = null;

                this.entity.setPhysicsEnabled(true);

                this.events.emit('playbackComplete');
            }
        }
    }
}