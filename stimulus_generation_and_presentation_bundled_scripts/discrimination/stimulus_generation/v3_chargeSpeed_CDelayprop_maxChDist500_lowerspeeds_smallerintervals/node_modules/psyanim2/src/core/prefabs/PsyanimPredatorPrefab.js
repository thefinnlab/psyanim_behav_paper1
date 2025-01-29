import PsyanimEntityPrefab from "../PsyanimEntityPrefab.js";

import PsyanimVehicle from "../components/steering/PsyanimVehicle.js";
import PsyanimArriveBehavior from "../components/steering/PsyanimArriveBehavior.js";
import PsyanimSeekBehavior from "../components/steering/PsyanimSeekBehavior.js";
import PsyanimWanderBehavior from "../components/steering/PsyanimWanderBehavior.js";
import PsyanimFOVSensor from "../components/physics/PsyanimFOVSensor.js";
import PsyanimBasicPredatorBehavior from "../components/steering/PsyanimBasicPredatorBehavior.js";
import PsyanimPredatorAgent from "../components/steering/agents/PsyanimPredatorAgent.js";

import PsyanimFOVRenderer from "../components/rendering/PsyanimFOVRenderer.js";

/**
 *  Prefab for creating a `Predator Agent`.
 * 
 *  A `Predator Agent` will wander until the target comes within its field-of-view and remains within the `boredom distance`,
 *  at which point it will begin to chase the target in a direction offset by the `subtlety` parameter.
 * 
 *  A `Predator Agent` has the following components: 
 * 
 *  `PsyanimVehicle`, `PsyanimArriveBehavior`, `PsyanimSeekBehavior`, `PsyanimWanderBehavior`, 
 *  `PsyanimFOVSensor`, `PsyanimBasicPredatorBehavior`, `PsyanimPredatorAgent`
 */
export default class PsyanimPredatorPrefab extends PsyanimEntityPrefab {

    /** Predator Params */

    /**
     *  Maximum number of degrees which this agent will offset its chase direction.
     *  @type {Number}
     */
    subtlety;

    /**
     *  Determines how often the agent will recompute its chase direction offset based on the `subtlety` parameter.
     *  @type {Number}
     */
    subtletyLag;

    /**
     *  Max. distance, in 'px', to target beyond which this predator will no longer chase it
     *  @type {Number}
     */
    boredomDistance;

    /**
     *  If true, agent state transitions will be written out to Psyanim Debug Logs
     *  @type {boolean}
     */
    showDebugLogs;

    /**
     *  Toggles debug graphics for field-of-view.
     *  @type {boolean}
     */
    showDebugGraphics;

    /**
     *  The minimum amount of time the agent must wander before it can transition to pursuing the target.
     * 
     *  This is helpful for interactive scenes where we don't want the agent constantly pursuing a target.
     *  @type {Number}
     */
    minimumWanderTime;

    /** Field-of-view Params */

    /**
     *  Determines how whether or not this agent will have a limited field of view.
     * 
     *  If false, the agent's field of view will be 360 degrees and have unlimited range.
     *  @type {boolean}
     */
    useFov;

    /**
     *  Field-of-view cone angle, in degrees.
     *  @type {Number}
     */
    fovAngle;

    /**
     *  Field-of-view resolution, in degrees.  
     * 
     *  This determines how many degrees of the field-of-view an entity must occupy to be detected.
     * 
     *  The lower the number, the higher the resolution, and the more CPU overhead this component has.
     * 
     *  @type {Number}
     */
    fovResolution;

    /** Chase Params */

    /**
     *  Maximum speed at which this agent will chase the target.
     *  @type {Number}
     */
    maxChaseSpeed;

    /**
     *  Maximum acceleration at which this agent will reach during a chase.
     *  @type {Number}
     */
    maxChaseAcceleration;

    /**
     *  Distance, in px, from target which agent will come to rest during a chase.
     *  @type {Number}
     */
    chaseInnerDecelerationRadius;

    /**
     *  Distance, in px, from target which the agent will begin slowing down during a chase.
     *  @type {Number}
     */
    chaseOuterDecelerationRadius;

    /** Wander Params */

    /**
     *  Radius of the wander circle
     *  @type {Number}
     */
    wanderRadius;

    /**
     *  Distance the wander circle is offset from the agent's position.
     *  @type {Number}
     */
    wanderOffset;

    /**
     *  Maximum number of degrees the wander target can move around the wander circle per frame.
     */
    maxWanderAngleChangePerFrame;

    /**
     *  Maximum speed at which the agent can wander.
     *  @type {Number}
     */
    maxWanderSpeed;

    /**
     *  Maximum rate at which the agent can accelerate when wandering.
     *  @type {Number}
     */
    maxWanderAcceleration;

    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        super(shapeParams, matterOptions);

        // predator params
        this.subtlety = 30;
        this.subtletyLag = 500;
        this.boredomDistance = 500;
        this.showDebugLogs = false;
        this.showDebugGraphics = false;

        // fov params
        this.useFov = false;
        this.fovAngle = 120;
        this.fovResolution = 5;

        // chase params
        this.maxChaseSpeed = 1.5;
        this.maxChaseAcceleration = 0.1;
        this.chaseInnerDecelerationRadius = 16;
        this.chaseOuterDecelerationRadius = 40;

        // wander params
        this.wanderRadius = 50;
        this.wanderOffset = 250;
        this.maxWanderAngleChangePerFrame = 10;
        this.maxWanderSpeed = 1.5;
        this.maxWanderAcceleration = 0.1;
        this.minimumWanderTime = 0;
    }

    create(entity) {

        super.create(entity);

        let vehicle = entity.addComponent(PsyanimVehicle);

        let arrive = entity.addComponent(PsyanimArriveBehavior);
        arrive.maxSpeed = this.maxChaseSpeed;
        arrive.maxAcceleration = this.maxChaseAcceleration;
        arrive.innerDecelerationRadius = this.chaseInnerDecelerationRadius;
        arrive.outerDecelerationRadius = this.chaseOuterDecelerationRadius;

        let seek = entity.addComponent(PsyanimSeekBehavior);
        seek.maxSpeed = this.maxWanderSpeed;
        seek.maxAcceleration = this.maxWanderAcceleration;

        let wander = entity.addComponent(PsyanimWanderBehavior);
        wander.seekBehavior = seek;
        wander.radius = this.wanderRadius;
        wander.offset = this.wanderOffset;
        wander.maxWanderAngleChangePerFrame = this.maxWanderAngleChangePerFrame;

        let fovSensor = null;

        if (this.useFov)
        {
            fovSensor = entity.addComponent(PsyanimFOVSensor);
            fovSensor.fovAngle = this.fovAngle;
            fovSensor.fovRange = this.boredomDistance;
            fovSensor.resolution = this.fovResolution;    
        }

        if (this.useFov && this.showDebugGraphics)
        {
            let fovRenderer = entity.addComponent(PsyanimFOVRenderer);
            fovRenderer.fovSensor = fovSensor;
        }

        let predator = entity.addComponent(PsyanimBasicPredatorBehavior);
        predator.arriveBehavior = arrive;
        predator.wanderBehavior = wander;
        predator.fovSensor = fovSensor;
        predator.subtlety = this.subtlety;
        predator.subtletyLag = this.subtletyLag;
        predator.boredomDistance = this.boredomDistance;
        predator.debug = this.showDebugLogs;
        predator.minimumWanderTime = this.minimumWanderTime;

        let predatorAgent = entity.addComponent(PsyanimPredatorAgent);
        predatorAgent.vehicle = vehicle;
        predatorAgent.predatorBehavior = predator;

        return entity;
    }
}