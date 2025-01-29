import Phaser from 'phaser';

import PsyanimConstants from '../../../PsyanimConstants.js';
import PsyanimComponent from '../../../PsyanimComponent.js';

import PsyanimPath from '../../../utils/PsyanimPath.js';

export default class PsyanimPathFollowingAgent extends PsyanimComponent {

    arriveAgent;

    currentPathVertices;

    targetPositionOffset;

    reverse;

    loop;

    constructor(entity) {

        super(entity);

        this.currentPathVertices = [];
        this.targetPositionOffset = 50;
        this.loop = false;
        this.reverse = true;

        this._pathFollowingTarget = this.scene.addEntity(this.entity.name + '_pathFollowingTarget', 0, 0, 
            { isEmpty: true },
            { collisionFilter: PsyanimConstants.DEFAULT_VISUAL_ONLY_COLLISION_FILTER });

        this._direction = 1;
    }

    setupArriveAgent() {

        this.arriveAgent.enabled = true;
        this.arriveAgent.target = this._pathFollowingTarget;
    }

    onEnable() {

        super.onEnable();

        this.setupArriveAgent();
    }

    onDisable() {

        super.onDisable();

        this.arriveAgent.enabled = false;
    }

    _convertMatterVerticesToPhaserVector2(vertices) {

        let phaserVerts = [];

        for (let i = 0; i < vertices.length; ++i)
        {
            phaserVerts.push(new Phaser.Math.Vector2(
               vertices[i].x, vertices[i].y 
            ));
        }

        return phaserVerts;
    }

    afterCreate() {

        super.afterCreate();

        this.currentPathVertices = this._convertMatterVerticesToPhaserVector2(this.currentPathVertices);

        this._currentPath = new PsyanimPath(this.currentPathVertices);

        this.targetParameterOffset = this.targetPositionOffset / this._currentPath.getTotalLength();

        this.setupArriveAgent();
    }

    _computePathFollowingTargetPosition() {

        let parameter = this._currentPath.getParameter(this.entity.position);

        if (this.reverse)
        {
            if (this._direction > 0 && (1.0 - parameter) < 0.01)
            {
                this._direction *= -1;
            }
            else if (this._direction < 0 && parameter < 0.01)
            {
                this._direction *= -1;
            }
        }

        let targetParameter = parameter + this.targetParameterOffset * this._direction;

        if (targetParameter < 0)
        {
            targetParameter = 0;
        }
        else if (targetParameter > 1)
        {
            targetParameter = 1;
        }

        let targetPosition = this._currentPath.getPosition(targetParameter);

        this._pathFollowingTarget.x = targetPosition.x;
        this._pathFollowingTarget.y = targetPosition.y;
    }

    update(t, dt) {

        super.update(t, dt);

        this._computePathFollowingTargetPosition();
    }
}